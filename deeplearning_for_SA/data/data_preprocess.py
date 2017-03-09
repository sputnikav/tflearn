"""
Pre-processing data to feed into LSTM.


- Sources:
        + fine_text
        +
- Output:
        + ready_text: labeled text that contains existed-in-Sentiment words


Author: Thanh L.X.

"""

import numpy as np
# import pandas as pd
# from scipy import stats, integrate
# import matplotlib.pyplot as plt

# plot distribution of sentences-length
import seaborn as sns
import pickle


# function to list sentiment word
def file_to_list(input):
    source = open(input, "r")
    list_Dict = []
    for line in source:
        line = line.strip()
        word = line.split()[0]
        # word = line.split("###")[0]
        list_Dict.append(word)
    return list_Dict
    source.close()


# function to filter text with words from dictionary
def filtering(input, output, output_empty, list_dict, id_features):
    source = open(input, "r")
    output = open(output, "w")
    output_empty = open(output_empty, "w")
    count_line = 0
    for line in source:
        line = line.strip()
        if line:
            count_line += 1
            line = line.split("###")
            word = line[0]
            output.write(word)
            output.write("###")
            content = line[1]
            strin = content.split()
            check = 1
            for go in strin:
                if go in list_dict:
                    check = 0
                    output.write(go)
                    output.write(" ")
            if check:
                output_empty.write(word)
                output_empty.write("###")
                output_empty.write(content)
                output_empty.write("\n")
            else:
                id_features.append(count_line)
            output.write("\n")
    source.close()
    output.close()
    output_empty.close()


# function to extract additional features
def extract_features(input, output, id_features):
    source = open(input, "r")
    output = open(output, "w")
    count_line = 0
    for line in source:
        line = line.strip()
        if line:
            count_line+=1
            if count_line in id_features:
                context = line.split()[-4:]
                for feat in context:
                    output.write(str(feat))
                    output.write(" ")
                output.write("\n")
    source.close()
    output.close()


# split features into training/test
def split_features(input, out_train, out_test, train_length):
    source = open(input, "r")
    train_out = open(out_train, "w")
    test_out = open(out_test, "w")
    count_line = 0
    for line in source:
        line = line.strip()
        if line:
            count_line += 1
            if count_line < train_length+1:
                for feat in line.split():
                    feat = feat.split(":")[1]
                    train_out.write(feat)
                    train_out.write(" ")
                train_out.write("\n")
            else:
                for feat in line.split():
                    feat = feat.split(":")[1]
                    test_out.write(feat)
                    test_out.write(" ")
                test_out.write("\n")
    source.close()
    train_out.close()
    test_out.close()


# function to filter out the empty line
def summarize(input,output):
    source = open(input, "r")
    output = open(output, "w")
    for line in source:
        line = line.strip()
        content = line.split("###")[1]
        if content:
            output.write(line)
            output.write("\n")
    source.close()
    output.close()


# function to check len
def check_len(input):
    source = open(input, "r")
    list_length = []
    for line in source:
        line = line.strip()
        if line:
            content = line.split("###")[1]
            list_length.append(len(content))
    return list_length


# function oto check label
def check_label(input):
    source = open(input, "r")
    list_label = []
    for line in source:
        line = line.strip()
        if line:
            word = line.split("###")[0]
            list_label.append(int(word))
    return list_label


# function to visualize distribution:
def visualize(data, x_name, y_name, title_name):
    axis = sns.distplot(data)
    sns.set(color_codes=True)
    # axis.set(xlabel='length', ylabel='distribution', title='length-distribution of sentence-representation')
    axis.set(xlabel=x_name, ylabel=y_name, title=title_name)
    # axis.text(0, 1, 'Left the plot', fontsize = 20, rotation=90)
    # axis.text(1.02, 1, 'Right the plot', fontsize = 20, rotation=270)
    sns.plt.show()


# function to split data into Train/Test

def split_data(input, train_data, train_label, test_data, test_label, train_length, list_Dict):
    source = open(input, "r")
    data_train = open(train_data, "w")
    label_train = open(train_label, "w")
    data_test = open(test_data, "w")
    label_test = open(test_label, "w")
    count_line = 0
    for line in source:
        line = line.strip()
        if line:
            count_line += 1
            label = line.split("###")[0]
            strin = line.split("###")[1]
            if count_line < train_length+1:
                label_train.write(label)
                for go in strin.split():
                    idx = list_Dict.index(go)
                    data_train.write(str(idx))
                    data_train.write(" ")
                label_train.write("\n")
                data_train.write("\n")
            else:
                label_test.write(label)
                for go in strin.split():
                    idx = list_Dict.index(go)
                    data_test.write(str(idx))
                    data_test.write(" ")
                label_test.write("\n")
                data_test.write("\n")
    source.close()
    data_train.close()
    data_test.close()
    label_train.close()
    label_test.close()


# load Dictionary and save to outputname
def one_hot_table(filename, length_dict, length_class):
    source = open(filename, "r")
    # list_class = np.empty(length_dict)
    embedding_class = np.arange(length_dict)
    count_line = 0
    for line in source:
        line = line.strip()
        if line:
            embedding_class[count_line] = int(line.split()[1])
            count_line += 1
    source.close()
    table_dict = np.zeros((length_dict, length_dict))
    table_class = np.zeros((length_dict, length_class))

    embedding_dict = np.arange(length_dict)

    table_dict[embedding_dict, embedding_dict] = 1
    # print (Table_Dict)
    # print (Table_Dict.shape)

    table_class[embedding_dict, embedding_class] = 1
    # print (Table_Class[:10])
    # print (Table_Class.shape)

    table = np.concatenate((table_dict, table_class), axis=1)
    # print (Table[:3])
    # print (Table.shape)

    # save_to_pickle("Table.pickle", Table)
    return table


def save_to_pickle(filename, obj):
    with open(filename, "w") as f:
        pickle.dump(obj, f)


def load_from_pickle(filename):
    with open(filename) as f:
        return pickle.load(f)

if __name__ == "__main__":

    # list file-name
    dict = "Dictionary_list"
    source = "fine_text"
    # filtered_text = "filtered_text"
    filtered_text = "filtered_text_cloned"
    # empty_text = "empty_text"
    empty_text = "empty_text_cloned"
    ready_text = "ready_text"
    train_data = "train_data_SA"
    train_label = "train_label_SA"
    test_data = "test_data_SA"
    test_label = "test_label_SA"
    train_length = 5000

    # additional features
    source_features = "SVM_features"
    ready_features = "add_features"
    id_features = []
    train_feat = "train_add_SA"
    test_feat = "test_add_SA"

    # read Dictionary to list
    print "read Dictionary to list"
    list_Dict = file_to_list(dict)

    # save_to_pickle("dict.pickle", list_Dict)

    # filter the fine-text
    print "filter the fine-text"
    filtering(source, filtered_text, empty_text, list_Dict, id_features)

    extract_features(source_features, ready_features, id_features)

    split_features(ready_features, train_feat, test_feat, train_length)

    """
    summarize(filtered_text, ready_text)

    # check distribution of sentence-length
    list_Length = check_len(ready_text)
    print list_Length[:10]
    print "max length: " + str(max(list_Length))
    print "min length: " + str(min(list_Length))

    visualize(list_Length, x_name="length", y_name="distribution", title_name="length-distribution "
                                                                              "of sentence-representation")

    # split data -> Training/test

    list_Dict = load_from_pickle("dict.pickle")

    print "splitting data"
    split_data(ready_text, train_data, train_label, test_data, test_label, train_length, list_Dict)

    # one-hot table
    length_dict = 6988
    length_class = 7
    Table = one_hot_table("Dictionary_list_fixed", length_dict, length_class)
    """
    # check distribution of label of non-sentiment-word sentences
    # list_label = check_label(empty_text)
    # visualize(list_label, x_name="label", y_name="estimated PDF", title_name="label-distribution of empty text")

    """
    label_list = file_to_list(empty_text)
    print (label_list)
    print (label_list.count("1"))
    print (label_list.count("2"))
    print (label_list.count("3"))
    print (label_list.count("4"))
    """
    # list_label = check_label(source)
    # visualize(list_label, x_name="label", y_name="estimated PDF", title_name="label-distribution of source text")

    print "Done."
