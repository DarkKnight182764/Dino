import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_one_hot(s, tokens):
    res = np.zeros((1, len(s), len(tokens)))
    for ic, char in enumerate(s):
        if char == " ":
            continue
        res[0, ic, tokens.index(char)] = 1
    return res


def get_char(one_hot, tokens):
    s = ""
    # print(one_hot)
    if len(one_hot.shape) == 2:
        for row in one_hot:
            # print(row.shape)
            row = list(row)
            # print(sum(row))
            s += tokens[row.index(max(row))]
    if len(one_hot.shape) == 1:
        one_hot = list(one_hot)
        s += tokens[one_hot.index(max(one_hot))]
    return s


if __name__ == '__main__':
    m = 0
    filename = "dinos.txt"
    with open(filename) as f:
        chars = f.read()
        tokens = list(set(chars))
        tokens = sorted(tokens)

    with open(filename) as f:
        names = f.readlines()


    def gen():
        while True:
            for i, name in enumerate(names):
                name = name.strip()
                name += "\n"
                y = get_one_hot(name, tokens)

                name = name.strip()
                name = " " + name
                X = get_one_hot(name, tokens)
                yield (X, y)


    gru = keras.layers.GRU(256, input_dim=len(tokens), return_sequences=True, activation=tf.nn.tanh,
                           stateful=True, batch_size=1, name="gru")
    model = keras.Sequential()
    model.add(gru)
    model.add(keras.layers.Dense(len(tokens), activation="softmax", name="dense_op"))
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="categorical_crossentropy")
    model.fit_generator(gen(), 1536, epochs=40)

    # model.save("e40dinos.h5")
    # model = keras.models.load_model("e40dinos.h5")

    while True:
        s = input("Starting char")
        n = s
        model.reset_states()
        p = model.predict(get_one_hot(s, tokens))
        # print("p", p.shape)
        while True:
            # print(lstm.states[0].numpy())
            ch = get_char(np.squeeze(p, 0), tokens)
            print(ch)
            if (ch == "\n"):
                print("Final name:" + n)
                break
            n += ch
            p = np.squeeze(p, (0, 1))
            nex = np.random.choice(len(tokens), p=p)
            # print("nex:", nex)
            p = get_one_hot(tokens[nex], tokens)
            p = model.predict(p)
            t = input()
            if t == "n":
                print("Final name:" + n)
                break
