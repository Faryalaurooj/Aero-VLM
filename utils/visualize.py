import matplotlib.pyplot as plt

def show_image(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

