import re
import tensorflow as tf

log_file = "train_cout.log"
logdir = "tb_logs"

writer = tf.summary.create_file_writer(logdir)

epoch_pattern = re.compile(r"Epoch (\d+)/\d+")

with open(log_file) as f:
    lines = f.readlines()

epoch = None

for line in lines:

    m = epoch_pattern.search(line)
    if m:
        epoch = int(m.group(1))
        continue

    if "step -" in line:
        parts = line.strip().split(" - ")[-1].split(" - ")

        metrics = {}
        for p in parts:
            k, v = p.split(": ")
            metrics[k] = float(v)

        with writer.as_default():
            for k, v in metrics.items():
                tf.summary.scalar(k, v, step=epoch)

writer.close()
