from repvgg import create_RepVGG_A0
import torch

def main():
    input = torch.rand(4, 3, 713, 713)
    vgg = create_RepVGG_A0(deploy=False)

    out = vgg(input)

    print(out)

if __name__ == '__main__':
    main()