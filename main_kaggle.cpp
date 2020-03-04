// https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch?scriptVersionId=8935337
#include <torch/script.h>
#include <torch/torch.h>

#include <random>
#include <string>

#include "common.h"

const char* NET_FILENAME = CMAKE_CURRENT_SOURCE_DIR "/charrnn_net_dante.pt";

struct TensorXY
{
    torch::Tensor x, y;
};

struct GetBatches
{
    int n = 0;
    int batch_size;
    int seq_length;
    int n_batches;
    VI arr;
    // Create a generator that returns batches of size
    // batch_size x seq_length from arr.
    GetBatches(VI arr_, int batch_size, int seq_length)
        : batch_size(batch_size), seq_length(seq_length), arr(move(arr_))
    {
        // Arguments
        //---------
        // arr: Array you want to make batches from
        // batch_size: Batch size, the number of sequences per batch
        // seq_length: Number of encoded chars in a sequence
        auto batch_size_total = batch_size * seq_length;
        n_batches = (~arr - 1) / batch_size_total;
    }
    optional<TensorXY> next()
    {
        if (n >= seq_length * n_batches) {
            return {};
        }

        auto x =
            torch::zeros({batch_size, seq_length}, torch::TensorOptions().dtype(torch::kInt64));
        auto y =
            torch::zeros({batch_size, seq_length}, torch::TensorOptions().dtype(torch::kInt64));
        FOR (rix, 0, < batch_size) {
            auto lx0 = rix * seq_length * n_batches + n;
            FOR (cix, 0, < seq_length) {
                x[rix][cix] = arr[lx0 + cix];
                y[rix][cix] = arr[lx0 + cix + 1];
            }
        }

        n += seq_length;

        return TensorXY{x, y};
    }
};
/*
auto int2char = [chars](int i) {};
FOR (i, 0, < ~chars) {
    char2intmap[chars[i]] = i;
}
auto char2int = [char2intmap](char c) {
    auto it = char2intmap.find(c);
    assert(it != char2intmap.end());
    return it->second;
};
*/

class CharMap
{
    using Char2IntMap = array<int, CHAR_MAX - CHAR_MIN + 1>;

    const vector<char> chars;
    const Char2IntMap char2intmap;

    static int char2arrayix(char c) { return c - CHAR_MIN; }
    static char arrayix2char(int i) { return i + CHAR_MIN; }

    static Char2IntMap create_char2intmap(const vector<char>& chars)
    {
        Char2IntMap char2intmap;
        char2intmap.fill(-1);
        FOR (i, 0, < ~chars) {
            char2intmap[char2arrayix(chars[i])] = i;
        }
        return char2intmap;
    }

public:
    CharMap(vector<char> chars) : chars(move(chars)), char2intmap(create_char2intmap(this->chars))
    {}
    int char2int(char c) const
    {
        int ix = char2arrayix(c);
        assert_between_co(ix, 0, ~char2intmap);
        auto i = char2intmap[ix];
        assert(i >= 0);
        return i;
    }
    int int2char(int i) const
    {
        assert_between_co(i, 0, ~chars);
        return chars[i];
    }
    int size() const { return ~chars; }
};

struct CharRNN : torch::nn::Module
{
    const double drop_prob;
    const int n_layers;
    const int n_hidden;
    const double lr;
    const CharMap charmap;
    torch::nn::LSTM lstm = nullptr;
    torch::nn::Dropout dropout = nullptr;
    torch::nn::Linear fc = nullptr;

    CharRNN(CharMap charmap_arg,
            int n_hidden = 612,
            int n_layers = 4,
            double drop_prob = 0.5,
            double lr = 0.001)
        : drop_prob(drop_prob),
          n_layers(n_layers),
          n_hidden(n_hidden),
          lr(lr),
          charmap(move(charmap_arg))
    {
        // creating character dictionaries
        auto options =
            torch::nn::LSTMOptions(~charmap, n_hidden).layers(n_layers).batch_first(true);
        if (drop_prob > 0) {
            options.dropout(drop_prob);
        }
        lstm = register_module("lstm", torch::nn::LSTM(options));
        dropout = register_module("dropout", torch::nn::Dropout(drop_prob));
        fc = register_module("fc", torch::nn::Linear(n_hidden, ~charmap));
    }
    // These inputs are x, and the hidden/cell state `hidden`.
    torch::nn::RNNOutput forward(const torch::Tensor& x, torch::Tensor hidden)
    {
        auto rnn_output = lstm->forward(x, hidden);
        auto out = dropout(rnn_output.output);
        // Stack up LSTM outputs using view
        // you may need to use contiguous to reshape the output
        out = out.contiguous().view({-1, n_hidden});
        out = fc(out);
        return {out, rnn_output.state};
    }
    torch::Tensor init_hidden(int batch_size)
    {
        // Create two new tensors with sizes n_layers x batch_size x n_hidden,
        // initialized to zero, for hidden state and cell state of LSTM
        // auto& weight = this->parameters().front();
        // torch::TensorOptions to;
        // return weight.new_zeros({2, n_layers, batch_size, n_hidden},to.device("cpu"));
        return torch::zeros({2, n_layers, batch_size, n_hidden});
    }
};

// net: CharRNN network
// data: text data to train the network
// epochs: Number of epochs to train
// batch_size: Number of mini-sequences per mini-batch, aka batch size
// seq_length: Number of character steps per mini-batch
// lr: learning rate
// clip: gradient clipping
// val_frac: Fraction of data to hold out for validation
// print_every: Number of steps for printing training and validation loss
struct TrainArgs
{
    int epochs = 10;
    int batch_size = 10;
    int seq_length = 50;
    double lr = 0.001;
    int clip = 5;
    double val_frac = 0.1;
    int print_every = 10;
};

void train(shared_ptr<CharRNN> pnet, const VI& full_data, TrainArgs ta)
{
    auto& net = *pnet;
    const auto [epochs, batch_size, seq_length, lr, clip, val_frac, print_every] = ta;
    net.train();
    auto opt = torch::optim::Adam(net.parameters(), lr);
    auto criterion = torch::nn::CrossEntropyLoss();

    // create training and validation data
    auto val_idx = int(~full_data * (1 - val_frac));
    auto data = VI{BE_BN(full_data, val_idx)};
    auto val_data = VI{BE_XE(full_data, val_idx)};

    int counter = 0;
    int n_chars = ~net.charmap;
    FOR (e, 0, < epochs) {
        // initialize hidden state
        auto h = net.init_hidden(
            batch_size);  // zeroed out?? we could use the implicit default arg, couldn't we?

        GetBatches gb(data, batch_size, seq_length);
        while (auto xy = gb.next()) {
            ++counter;
            // One-hot encode our data and make them Torch tensors
            auto inputs = torch::nn::functional::one_hot(xy->x, n_chars).toType(torch::kFloat);
            auto targets = xy->y;
            // Creating new variables for the hidden state, otherwise
            // we'd backprop through the entire training history
            h = h.data();  //???
            net.zero_grad();
            // get the output from the model
            auto output_h = net.forward(inputs, h);
            auto output = output_h.output;
            h = output_h.state;

            // calculate the loss and perform backprop
            auto loss = criterion(output, targets.view(batch_size * seq_length));
            loss.backward();
            // `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch::nn::utils::clip_grad_norm_(net.parameters(), clip);
            opt.step();

            // loss stats
            if (counter % print_every == 0) {
                // torch::save(pnet, NET_FILENAME);
                // Get validation loss
                auto val_h = net.init_hidden(batch_size);
                vector<double> val_losses;
                net.eval();
                GetBatches gb(val_data, batch_size, seq_length);
                while (auto xy = gb.next()) {
                    // One - hot encode our data and make them Torch tensors
                    auto inputs =
                        torch::nn::functional::one_hot(xy->x, n_chars).toType(torch::kFloat);
                    auto targets = xy->y;

                    // Creating new variables for the hidden state, otherwise
                    // we'd backprop through the entire training history
                    val_h = val_h.data();  //??

                    auto ovh = net.forward(inputs, val_h);
                    auto val_loss = criterion(ovh.output, targets.view(batch_size * seq_length));
                    val_losses.PB(val_loss.item<double>());
                }
                net.train();
                printf("Epoch: %d/%d...", e + 1, epochs);
                printf("Step: %d...", counter);
                printf("Loss: %f...", loss.item<double>());
                printf("Val Loss: %f\n", mean(BE(val_losses)));
            }
        }
    }
}

void show_tensor(const char* title, torch::Tensor t0)
{
    auto ss0 = t0.sizes();
    printf("%s(%s)", title, to_string(BE(ss0)).c_str());
    auto t = t0.squeeze();
    auto ss = t.sizes();
    if (~ss == 0) {
        printf(" = [%f]\n", t.item<double>());
    } else if (~ss == 1) {
        printf(" = [");
        FOR (i, 0, < min(ss[0], 62LL)) {
            printf("%f ", t[i].item<double>());
        }
        printf("]\n");
    } else if (~ss == 2) {
        printf(" = [\n");
        FOR (r, 0, < min(ss[0], 20LL)) {
            printf("\t");
            FOR (c, 0, < min(ss[0], 62LL)) {
                printf("%f ", t[r][c].item<double>());
            }
            printf("\n");
        }
        printf("]\n");
    } else {
        printf("\n");
    }
}

// Given a character, predict the next character.
// Returns the predicted character and the hidden state.
tuple<char, torch::Tensor> predict(CharRNN& net,
                                   char chr,
                                   torch::Tensor h_arg = {},
                                   maybe<int> top_k = {},
                                   torch::jit::script::Module module = {})
{
    // tensor inputs
    auto x = torch::tensor({{net.charmap.char2int(chr)}});
    auto inputs = torch::one_hot(x, ~net.charmap).toType(torch::kFloat);

    show_tensor("inputs hot: ", inputs);

    // detach hidden state from history
    if (h_arg.has_storage()) {
        h_arg = h_arg.data();  //?
    }
    // get the output of the model
    torch::Tensor out;
    if (module._ivalue()) {
        assert(h_arg.size(0) == 2);
        auto tuple_h_c = torch::ivalue::Tuple::create(h_arg[0], h_arg[1]);
        vector<torch::IValue> forward_input_vec{inputs, tuple_h_c};
        auto tuple_out_h = module.forward(forward_input_vec);
        const auto& vec_out_h = tuple_out_h.toTuple()->elements();
        out = vec_out_h[0].toTensor();
        const auto& vec_h_c_next = vec_out_h[1].toTuple()->elements();
        auto state_h = vec_h_c_next[0].toTensor().unsqueeze(0);
        auto state_c = vec_h_c_next[1].toTensor().unsqueeze(0);
        h_arg = torch::cat({state_h, state_c}, 0);
    } else {
        auto out_h = net.forward(inputs, h_arg);
        out = out_h.output;
        h_arg = out_h.state;
    }
    // get the character probabilities
    // apply softmax to get p probabilities for the likely next character giving x
    auto p =
        torch::nn::functional::softmax(out, torch::nn::functional::SoftmaxFuncOptions(1));

    show_tensor("p: ", p);

    // get top characters
    // considering the k most probable characters with topk method
    torch::Tensor top_ch;

    if (!top_k) {
        top_ch = torch::tensor(mk_iota_0n(~net.charmap));
    } else {
        tie(p, top_ch) = p.topk(*top_k);
        top_ch = top_ch.squeeze();
    }

    // select the likely next character with some element of randomness
    p = p.squeeze();
    p /= p.sum();
    p = p.cumsum(0);

    auto r = torch::rand(1).item<double>();

    // FOR(i,0,<p.size(0)){
    //  printf("`%c`: %.4f\n",
    //}

    optional<int> result;
    if (top_k == 1) {
        assert(top_ch.dim() == 0);
        result = top_ch.item<int>();
    } else {
        FOR (i, 0, < p.size(0)) {
            if (r <= p[i].item<double>() || i + 1 == p.size(0)) {
                result = top_ch[i].item<int>();
                break;
            }
        }
    }

    // return the encoded value of the predicted char and the hidden state
    return {net.charmap.int2char(*result), h_arg};
}

string sample(CharRNN& net,
              int size,
              const string& prime = "Il",
              maybe<int> top_k = {},
              torch::jit::script::Module module = {})
{
    net.eval();

    // First off, run through the prime characters
    auto chars = prime;
    auto h = net.init_hidden(1);
    char chr;
    for (auto ch : prime) {
        tie(chr, h) = predict(net, ch, h, top_k, module);
    }
    chars += chr;

    // Now pass in the previous character and get a new one
    FOR (ii, 0, < size) {
        tie(chr, h) = predict(net, chars.back(), h, top_k, module);
        chars += chr;
    }
    return chars;
}

string load_dante_text()
{
    auto f = ifstream(CMAKE_CURRENT_SOURCE_DIR "/dante.txt");
    return read_file(f);
}

const bool PRINT_TESTS = false;

CharMap make_charmap(const string& text)
{
    set<char> s(BE(text));
    return CharMap(vector<char>(BE(s)));
}

auto make_net(CharMap charmap)
{
    int n_hidden = 512;
    int n_layers = 4;
    return std::make_shared<CharRNN>(move(charmap), n_hidden, n_layers);
}

void load_python_save(const char* s, vector<torch::Tensor>& parameters)
{
    ifstream f(s);
    assert(f.good());
    auto lines = read_lines(f);
    auto pit = parameters.begin();
    FOR (i, 0, < ~lines) {
        if (i < 3) {
            continue;
        }
        auto& l = lines[i];
        assert(l == "new torch.FloatTensor");
        auto& kl = lines[++i];
        assert(starts_with(kl, "key "));
        auto key = kl.substr(4);
        auto& sl = lines[++i];
        assert(starts_with(sl, "size torch.Size(["));
        auto xs = split(sl.substr(17), ", ])");
        if (~xs == 1) {
            int nc = stoi(xs[0]);
            printf("-- %s (%d)\n", key.c_str(), nc);
            assert(pit != parameters.end());
            auto& t = *pit;
            ++pit;
            assert(t.sizes().size() == 1 && t.sizes()[0] == nc);
            xs = split(lines[++i], " ");
            assert(~xs == nc);
            FOR (c, 0, < nc) {
                t[c] = stod(xs[c]);
            }
        } else {
            assert(~xs == 2);
            int nr = stoi(xs[0]);
            int nc = stoi(xs[1]);
            printf("-- %s (%d x %d)\n", key.c_str(), nr, nc);
            assert(pit != parameters.end());
            auto& t = *pit;
            ++pit;
            assert(t.sizes().size() == 2 && t.sizes()[0] == nr && t.sizes()[1] == nc);
            FOR (r, 0, < nr) {
                xs = split(lines[++i], " ");
                assert(~xs == nc);
                FOR (c, 0, < nc) {
                    t[r][c] = stod(xs[c]);
                }
            }
        }
    }
    assert(pit == parameters.end());
    printf("loaded\n");
}

void load_python_save_and_sample()
{
    auto text = load_dante_text();
    auto net = make_net(make_charmap(text));
    torch::load(net, NET_FILENAME);
    auto vs = net->parameters();
    load_python_save(CMAKE_CURRENT_SOURCE_DIR "/a.txt", vs);
    auto s = sample(*net, 1000, "Nel ", 5);
    printf("%s\n", s.c_str());
    printf("-------------------\n");
    s = sample(*net, 800, "E disse ", 5);
    printf("%s\n", s.c_str());
}

void load_pt_and_sample()
{
    auto text = load_dante_text();
    auto net = make_net(make_charmap(text));
    auto module = torch::jit::load(CMAKE_CURRENT_SOURCE_DIR "/traced.pt");
    auto vs = net->parameters();
    load_python_save(CMAKE_CURRENT_SOURCE_DIR "/a.txt", vs);
    auto s = sample(*net, 1000, "Nel ", 1, module);
    printf("%s\n", s.c_str());
    printf("-------------------\n");
    s = sample(*net, 800, "E disse ", 5, module);
    printf("%s\n", s.c_str());
}

int main()
{
    load_pt_and_sample();
    return 0;

    auto text = load_dante_text();
    auto charmap = make_charmap(text);

    if (PRINT_TESTS) {
        printf("--- First 300 chars ---\n%s\n--- END ---\n", text.substr(0, 300).c_str());
    }

    // encode the text
    VI encoded;
    for (auto c : text) {
        encoded.PB(charmap.char2int(c));
    }

    if (PRINT_TESTS) {
        printf("--- First 100 encoded ---\n%s\n--- END ---\n",
               to_string(encoded.begin(), encoded.begin() + 100).c_str());

        auto batches = GetBatches(encoded, 8, 50);
        auto xy = batches.next();

        assert(xy);
        cout << xy->x.narrow(1, 0, 10) << endl;
        cout << xy->y.narrow(1, 0, 10) << endl;
    }

    auto net = make_net(charmap);

    cout << *net << endl;

    int batch_size = 64;
    int seq_length = 160;  // max length verses
    int n_epochs = 50;     // start smaller if you are just testing initial behavior

    TrainArgs ta;
    ta.epochs = n_epochs;
    ta.batch_size = batch_size;
    ta.seq_length = seq_length;
    ta.lr = 0.001;
    ta.print_every = 10;
    train(net, encoded, ta);

    return 0;
}