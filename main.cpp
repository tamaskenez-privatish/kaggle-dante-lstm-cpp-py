#include <torch/torch.h>

#include <random>
#include <string>

#include "common.h"

// level-0: 6814
// level-1: 6+8=14
// level-2: hat meg nyolc az tizennegy

const char* CCSD = CMAKE_CURRENT_SOURCE_DIR;

const int EPOCH_COUNT = 10;
const int N_BATCHES = 1000;
const int ONE_BATCH_SIZE = 100;
const int SEQ_LEN = 100;
const int HIDDEN_SIZE = 77;
const int RN_LAYERS = 3;
const int THE_LEVEL = 1;

enum L0Symbols
{
    L0_START = 10,
    L0_STOP,
    L0_LAST_SYMBOL = L0_STOP
};

enum L1Symbols
{
    L1_START = 10,
    L1_STOP,
    L1_PLUS,
    L1_EQUALS,
    L1_LAST_SYMBOL = L1_EQUALS
};

string int_to_word(int i)
{
    const char* words[] = {"nulla",   "egy",      "ketto",      "harom",      "negy",
                           "ot",      "hat",      "het",        "nyolc",      "kilenc",
                           "tiz",     "tizenegy", "tizenketto", "tizenharom", "tizennegy",
                           "tizenot", "tizenhat", "tizenhet",   "tizennyolc"};
    assert_between_cc(i, 0, 18);
    return words[i];
}
int encode_l2_char(char c)
{
    // ' ' -> 0
    // a .. z -> 1 .. 'z' - 'a'+1
    if (c == ' ') {
        return 0;
    }
    assert_between_cc(c, 'a', 'z');
    return c - 'a' + 1;
}
enum L2Symbols
{
    L2_START = 'z' - 'a' + 1 + 1,
    L2_STOP,
    L2_LAST_SYMBOL = L2_STOP
};

string to_string(const VI& v, int level)
{
    string r;
    switch (level) {
        case 0:
            for (auto i : v) {
                if (i == L0_START) {
                    r += "<START>";
                } else if (i == L0_STOP) {
                    r += "<STOP>";
                } else {
                    assert_between_cc(i, 0, 9);
                    r += to_string(i);
                }
            }
            break;
        case 1:
            for (auto i : v) {
                if (i == L1_START) {
                    r += "<START>";
                } else if (i == L1_STOP) {
                    r += "<STOP>";
                } else if (i == L1_PLUS) {
                    r += "+";
                } else if (i == L1_EQUALS) {
                    r += "=";
                } else {
                    assert_between_cc(i, 0, 9);
                    r += to_string(i);
                }
            }
            break;
        case 2:
            for (auto i : v) {
                if (i == L2_START) {
                    r += "<START>";
                } else if (i == L2_STOP) {
                    r += "<STOP>";
                } else if (i == 0) {
                    r += " ";
                } else {
                    assert_between_cc(i, 1, 'z' - 'a' + 1);
                    r += i + 'a' - 1;
                }
            }
            break;
        default:
            UNREACHABLE;
    }
    return r;
}

struct Generator
{
    const int level;
    const bool enable_over_nine;
    default_random_engine dre;
    uniform_int_distribution<> zero_to_nine{0, 9};

    explicit Generator(int level, bool enable_over_nine)
        : level(level), enable_over_nine(enable_over_nine)
    {}
    int one_hot_size() const
    {
        switch (level) {
            case 0:
                return L0_LAST_SYMBOL + 1;
            case 1:
                return L1_LAST_SYMBOL + 1;
            case 2:
                return L2_LAST_SYMBOL + 1;
            default:
                UNREACHABLE;
        }
        return 0;
    }
    int start_symbol() const
    {
        switch (level) {
            case 0:
                return L0_START;
            case 1:
                return L1_START;
            case 2:
                return L2_START;
            default:
                UNREACHABLE;
        }
        return 0;
    }
    int stop_symbol() const
    {
        switch (level) {
            case 0:
                return L0_STOP;
            case 1:
                return L1_STOP;
            case 2:
                return L2_STOP;
            default:
                UNREACHABLE;
        }
        return 0;
    }

    vector<int> next_plus(VI use_this_for_return, bool add_answer = true)
    {
        int x, y;
        x = zero_to_nine(dre);
        if (enable_over_nine) {
            y = zero_to_nine(dre);
        } else {
            y = uniform_int_distribution<>(0, 9 - x)(dre);
        }
        int z = x + y;
        VI r(move(use_this_for_return));
        r.clear();
        switch (level) {
            case 0:
                r.assign({L0_START, x, y});
                if (add_answer) {
                    if (z >= 10) {
                        r.PB(1);
                    }
                    r.insert(r.end(), {z % 10, L0_STOP});
                }
                break;
            case 1:
                r.assign({L1_START, x, L1_PLUS, y, L1_EQUALS});
                if (add_answer) {
                    if (z >= 10) {
                        r.PB(1);
                    }
                    r.insert(r.end(), {z % 10, L1_STOP});
                }
                break;
            case 2: {
                string s = int_to_word(x) + " meg " + int_to_word(y) + " az ";
                if (add_answer) {
                    s += int_to_word(z);
                }
                r = {L2_START};
                for (auto c : s) {
                    r.PB(encode_l2_char(c));
                }
                if (add_answer) {
                    r.PB(L2_STOP);
                }
            }
        }
        return r;
    }
};

#define USE_LSTM

// Define a new Module.
struct Net : torch::nn::Module
{
    const bool use_relu = false;
    Net(int one_hot_size, int hidden_size, int rn_layers)
    {
#ifdef USE_LSTM
        auto options = torch::nn::LSTMOptions(one_hot_size, hidden_size).layers(rn_layers);
        if (0) {
            options.dropout(0.5);
        }
        r1 = register_module("r1", torch::nn::LSTM(options));
#else
        r1 = register_module("r1", torch::nn::GRU(one_hot_size, hidden_size));
#endif
        l1 = register_module("l1", torch::nn::Linear(hidden_size, one_hot_size));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        auto rnn_out = r1->forward(x);
        cout << "1 " << rnn_out.state.sizes() << "\n---------------" << endl;
        auto ps = r1->parameters();
        printf("%d pars\n", ~ps);
        for(auto&t:ps){
            cout << "par " << t.sizes() << endl;
        }
        if (use_relu) {
            x = torch::relu(rnn_out.output);
        } else {
            x = rnn_out.output;
        }
        return l1->forward(x);
    }

    VI complete(torch::Tensor x, int stop_symbol, int max_length)
    {
        VI result;
        auto rnn_out = r1->forward(x);
        for (; ~result < max_length;) {
            torch::Tensor xx = rnn_out.output;
            if (use_relu) {
                xx = torch::relu(xx);
            }
            auto y = l1->forward(xx);
            auto last_y = y[y.size(0) - 1][0];
            int decoded = *last_y.argmax().data_ptr<int64_t>();
            // last_y one-hot-decode to next result
            result.PB(decoded);
            if (decoded == stop_symbol) {
                break;
            }
            // rnn_out.state and last_y becomes the next input for r1
            auto nextx = last_y.reshape({1, 1, -1});
            rnn_out = r1->forward(nextx, rnn_out.state);
        }
        return result;
    }

    // Use one of many "standard library" modules.
#ifdef USE_LSTM
    torch::nn::LSTM r1{nullptr};
#else
    torch::nn::GRU r1{nullptr};
#endif
    torch::nn::Linear l1{nullptr};
};

int main()
{
    torch::get_py
    Generator g(THE_LEVEL, false);

    // Create a new Net.
    auto net = std::make_shared<Net>(g.one_hot_size(), HIDDEN_SIZE, RN_LAYERS);
    printf("Made net\n");

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::Adam optimizer(net->parameters(), /*lr=*/0.001);
    printf("Made optimizer\n");
    VI tmp;
    FOR (epoch_ix, 0, < EPOCH_COUNT) {
        printf("epoch %d\n", epoch_ix);
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        auto loss_class = torch::nn::CrossEntropyLoss();
        VI sample_in, sample_out;
        FOR (batch_ix, 0, < N_BATCHES) {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::TensorOptions yopt = torch::TensorOptions().dtype(torch::kLong);
            torch::Tensor xs = torch::zeros({SEQ_LEN, ONE_BATCH_SIZE, g.one_hot_size()});
            torch::Tensor ys = torch::zeros({SEQ_LEN, ONE_BATCH_SIZE}, yopt);
            bool do_sample = sample_in.empty();
            FOR (seq_ix, 0, < ONE_BATCH_SIZE) {
                int next = 0;
                if (do_sample) {
                    sample_in.resize(SEQ_LEN);
                    sample_out.resize(SEQ_LEN);
                }
                while (next < SEQ_LEN) {
                    tmp = g.next_plus(tmp);
                    for (int i = 0; i < ~tmp && next < SEQ_LEN; ++i) {
                        auto x = tmp[i];
                        auto y = i + 1 < ~tmp ? tmp[i + 1] : tmp[0];
                        xs[next][seq_ix][x] = 1;
                        ys[next][seq_ix] = y;
                        if (do_sample) {
                            sample_in[next] = x;
                            sample_out[next] = y;
                        }
                        ++next;
                    }
                }
                do_sample = false;
            }
            torch::Tensor prediction = net->forward(xs);

            auto yhat = prediction.reshape({SEQ_LEN * ONE_BATCH_SIZE, g.one_hot_size()});
            auto gt = ys.reshape({SEQ_LEN * ONE_BATCH_SIZE});
            torch::Tensor loss = loss_class->forward(yhat, gt);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 10 == 0) {
                std::cout << "Epoch: " << epoch_ix << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
                {
                    if (0) {
                        VI seq;
                        FOR (k, 0, < 0) {
                            tmp = g.next_plus(tmp, true);
                            seq.insert(seq.end(), BE(tmp));
                        }
                        tmp = g.next_plus(tmp, false);
                        seq.insert(seq.end(), BE(tmp));
                        tmp = seq;
                        torch::Tensor xs = torch::zeros({~tmp, 1, g.one_hot_size()});

                        FOR (i, 0, < ~tmp) {
                            xs[i][0][tmp[i]] = 1;
                        }
                        if (1) {
                            auto answer = net->complete(xs, g.stop_symbol(), 2 * ~tmp);
                            printf("`%s` -> `%s`\n", to_string(tmp, g.level).c_str(),
                                   to_string(answer, g.level).c_str());
                        } else {
                            auto ys = net->forward(xs);
                            FOR (rix, 0, < ys.size(0)) {
                                auto row = ys[rix][0];
                                auto ps = torch::softmax(row, 0);
                                auto rs = to_string(VI(1, tmp[rix]), g.level);
                                printf("%8s ", rs.c_str());
                                FOR (cix, 0, < ps.size(0)) {
                                    auto f = ps[cix].item<float>();
                                    printf("%1.2f ", f);
                                }
                                printf("\n");
                            }
                            printf("\n");
                        }
                    } else {
                        printf("sample_in: `%s`\n", to_string(sample_in, g.level).c_str());
                        printf("sample_out: `%s`\n", to_string(sample_out, g.level).c_str());
                        sample_in.clear();
                        sample_out.clear();
                        torch::Tensor xs = torch::zeros({1, 1, g.one_hot_size()});
                        xs[0][0][g.start_symbol()] = 1;
                        auto answer = net->complete(xs, g.stop_symbol(), 33);
                        printf("`<START>` -> `%s`\n", to_string(answer, g.level).c_str());
                    }
                }
            }
        }
    }
}
