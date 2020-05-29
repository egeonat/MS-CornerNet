import os

import matplotlib.pyplot as plt

logs_dir = "C:/Users/batuy/Desktop/kaist_hg_ddp"


def main():
    focal = []
    pull = []
    push = []
    reg = []
    logs = os.listdir(logs_dir)
    for log_name in logs:
        if log_name[-4:] == ".txt":
            with open(logs_dir + "/" + log_name) as file:
                for line in file:
                    ind_focal = line.find("focal_loss")
                    if ind_focal != -1:
                        focal.append(float(line[ind_focal + 12: ind_focal + 19]))
                        ind_pull = line.find("pull_loss")
                        pull.append(float(line[ind_pull + 11: ind_pull + 18]))
                        ind_push = line.find("push_loss")
                        push.append(float(line[ind_push + 11: ind_push + 18]))
                        ind_reg = line.find("reg_loss")
                        reg.append(float(line[ind_reg + 10: ind_reg + 17]))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(focal)
    axs[0, 1].plot(reg)
    axs[1, 0].plot(pull)
    axs[1, 1].plot(push)
    plt.show()


if __name__ == "__main__":
    main()
