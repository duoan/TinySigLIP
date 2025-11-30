import pprint

import timm


def main():
    print("Hello from tinysiglip!")
    model_names = timm.list_models(pretrained=True)
    pprint.pprint(model_names)


if __name__ == "__main__":
    main()
