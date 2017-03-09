"""
Preparing for LSTM experiment on Social data

Source files:

0- Comparative
1- Decrement
2- Increment
3- Neg
4- Pos
5- PosNeg
6- reverse

Output file:

Dictionary_Sentiment

Format:
word .. class-label

"""


def read_file(file_input, file_output, state, position, class_id):
    source = open(file_input, "r")
    output = open(file_output, state)
    for line in source:
        line = line.strip()
        line = line.split()[position]
        output.write(line)
        output.write(" ")
        output.write(str(class_id))
        output.write("\n")
    source.close()
    output.close()

if __name__ == "__main__":
    """
    inp0 = "Comparative"
    inp1 = "Decrement"
    inp2 = "Increment"
    inp3 = "Neg"
    inp4 = "Pos"
    inp5 = "PosNeg"
    inp6 = "reverse"
    out = "Dictionary_Sentiment"
    read_file (inp0, out, "w", 0, 0)
    read_file (inp1, out, "a", 0, 1)
    read_file (inp2, out, "a", 0, 2)
    read_file (inp3, out, "a", 1, 3)
    read_file (inp4, out, "a", 1, 4)
    read_file (inp5, out, "a", 0, 5)
    read_file (inp6, out, "a", 0, 6)
    print "Done."
    """

    # dictionary check:
    """
    list_Dict = []
    source = open ("Dictionary_Sentiment","r")
    for line in source:
        line = line.strip()
        word = line.split()[0]
        list_Dict.append(word)
    source.close()
    print ("Number of words in Dictionary: ", len(set(list_Dict)))
    """

    # print out to file the
    from collections import defaultdict

    set_Dict = defaultdict(list)

    source = open("Dictionary_Sentiment", "r")
    for line in source:
        line = line.strip()
        word = line.split()[0]
        word_class = line.split()[1]
        set_Dict[word].append(word_class)
    source.close()

    """
    out = open("Dictionary_list", "w")
    for word in set_Dict:
        out.write(word)
        out.write(" ")
        for go in set_Dict[word]:
            out.write(go)
            out.write(" ")
        out.write("\n")
    out.close()
    """
    out = open("Dictionary_list_fixed", "w")
    for word in set_Dict:
        out.write(word)
        out.write(" ")
        out.write(set_Dict[word][0])
        out.write(" ")
        out.write("\n")
    out.close()

    print "Done."
