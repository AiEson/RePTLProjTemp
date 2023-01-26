import inspect
import os

# get now real dir
_NOW_DIR = os.path.dirname(inspect.getfile(inspect.currentframe()))


def main():
    train_name = input("New train name:")
    train_encoder = input(
        "Train Encoder(available at https://smp.readthedocs.io/en/latest/encoders.html)\nname:"
    )
    # ------------- Config the Replace Rule ------------------
    _REPLACE_CONTENT = {"{{train_name}}": train_name, "{{encoder_name}}": train_encoder}

    # Check the validation of the folder name.
    if os.path.isdir(train_name):
        raise f"Folder Exists({train_name}, please check the name input.)"

    # Create the new train dir.
    os.makedirs(os.path.join(_NOW_DIR, train_name))

    # List dir to get the template folder content.
    for file_name in os.listdir(os.path.join(_NOW_DIR, "template")):

        # Read Template File
        f = open(os.path.join(_NOW_DIR, "template", file_name), "r")
        content = f.read()
        f.close()

        # Do the Replace Rule
        for key, to in _REPLACE_CONTENT.items():
            content = content.replace(key, to)

        # Write the content to new file
        f_w = open(os.path.join(_NOW_DIR, train_name, file_name), "a")
        f_w.write(content)

        f.close()
        f_w.close()

    # --------------- Show Basic Info -------------------
    __import__("pprint").pprint(f"Create the new trainer `{train_name}` done!")
    __import__("pprint").pprint(
        f"PLEASE GO TO `./{train_name}/train.py` modify your Model!"
    )
    __import__("pprint").pprint("\nHere are the default start.sh content:\n")

    with open(os.path.join(_NOW_DIR, train_name, "start.sh")) as f:
        print(f.read())


if __name__ == "__main__":
    main()
