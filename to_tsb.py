import re
import os
from torch.utils.tensorboard import SummaryWriter

log_file = "train_cout.log"
tb_root = "tb_logs"

os.makedirs(tb_root, exist_ok=True)

writer = None
epoch = None

with open(log_file) as f:
    for line in f:

        # 新实验
        if line.startswith("# subject"):
            m = re.search(r"subject (.*), session (\d+)", line)
            subject = m.group(1)
            session = m.group(2)

            run_name = f"{subject}_session{session}"
            logdir = os.path.join(tb_root, run_name)

            if writer:
                writer.close()

            writer = SummaryWriter(logdir)
            print("New run:", run_name)

        # epoch
        elif line.startswith("Epoch"):
            epoch = int(line.split()[1].split("/")[0])

        # comment
        elif line.startswith(";"):
            continue

        # metrics
        elif " - " in line and "step" in line:
            parts = line.strip().split(" - ")

            metrics = {}
            for p in parts:
                if ":" in p:
                    k, v = p.split(":")
                    try:
                        metrics[k.strip()] = float(v.strip())
                    except:
                        pass

            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)

if writer:
    writer.close()
