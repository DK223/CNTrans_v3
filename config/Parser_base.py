import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default='CNTrans',
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--pretrained_dir", type=str, default="./resnet34-333f7ec4.pth",
                    help="Where to search for pretrained resnet models.")
parser.add_argument("--output_dir", default="output", type=str,
                    help="The output directory where checkpoints will be written.")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument("--img_size", default=224, type=int,
                    help="Resolution size")
parser.add_argument("--train_batch_size", default=16, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=16, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=15000, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--class_number", default=38, type=int,
                    help="the number of typhoon speed")
parser.add_argument("--eval_every", default=100, type=int,
                    help="Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--d_model', type=int, default=512,
                    help="length if model embedding vectors")
parser.add_argument('--hidden_dim', type=int, default=512,
                    help="length if model embedding vectors")
parser.add_argument('--max_position_embedding', type=int, default=1024,
                    help="max_position_embedding")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--dropout", default=0.1, type=float,
                    help="dropout rate")
parser.add_argument("--nheads", default=4, type=float,
                    help="number of heads")
parser.add_argument("--dim_feedforward", default=2048, type=float,
                    help="dim of feedforwards")
parser.add_argument("--enc_layers", default=2, type=int,
                    help="number of encoder layers")
parser.add_argument("--dec_layers", default=2, type=int,
                    help="dec_layers")
parser.add_argument("--pre_norm", default=True, type=bool,
                    help="normlization before attention or not")
parser.add_argument("--src_len", default=4, type=int,
                    help="length of input sequence")
parser.add_argument("--pos_emb_type", default='v4', type=str,
                    help="type of position_embedding")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
Args = parser.parse_args()
