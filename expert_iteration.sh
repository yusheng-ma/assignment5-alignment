    # parser.add_argument("--EI_num_G", type=int, default=5)
    # parser.add_argument("--SFT_num_epochs", type=int, default=1)
    # # n_expert_iteration = 5
    # parser.add_argument("--EI_batch_size", type=int, default=512)

uv run ./cs336_alignment/run_expert_iteration.py

uv run ./cs336_alignment/run_expert_iteration.py --EI_num_G 10

uv run ./cs336_alignment/run_expert_iteration.py --SFT_num_epochs 2

uv run ./cs336_alignment/run_expert_iteration.py --EI_batch_size 1024
uv run ./cs336_alignment/run_expert_iteration.py --EI_batch_size 2048