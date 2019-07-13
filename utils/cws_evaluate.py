from configs import cws_config


def cws_evaluete(real, mine):
    real_file = open(real, "r", encoding="utf-8")
    mine_file = open(mine, "r", encoding="utf-8")
    real_lines = real_file.readlines()
    mine_lines = mine_file.readlines()
    real_file.close()
    mine_file.close()
    total_real_words = 0
    total_mine_words = 0
    total_correct_words = 0
    for (real_line, mine_line) in zip(real_lines, mine_lines):
        real_line_list = real_line.strip().split(" ")
        mine_line_list = mine_line.strip().split(" ")
        total_real_words += len(real_line_list)
        total_mine_words += len(mine_line_list)
        i, j = 0, 0
        while i < len(real_line_list) and j < len(mine_line_list):
            if real_line_list[i] == mine_line_list[j]:
                total_correct_words += 1
                i += 1
                j += 1
            else:
                len1 = len(real_line_list[i])
                len2 = len(mine_line_list[j])
                while len1 != len2:
                    if len1 > len2:
                        j += 1
                        if j >= len(mine_line_list):
                            break
                        len2 += len(mine_line_list[j])
                    else:
                        i += 1
                        if i >= len(real_line_list):
                            break
                        len1 += len(real_line_list[i])
                i += 1
                j += 1
    precision = 1.00 * total_correct_words / total_mine_words
    recall = 1.00 * total_correct_words / total_real_words
    f1 = 2 / (1 / precision + 1 / recall)
    print("precision : %g, recall : %g, f1: %g " % (precision, recall, f1))


if __name__ == '__main__':
    config = cws_config.cws_config
    cws_evaluete(config['test'], config['test'] + "_seg2.txt")
