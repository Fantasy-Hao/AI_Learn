from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("imgs/dog.png")
print(img)

# ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
writer.add_image("Resize", img_resize)
print(img_resize.shape)

# Compose
trans_compose = transforms.Compose([trans_resize, trans_tensor])
img_compose = trans_compose(img)
writer.add_image("Compose", img_compose)

# RandomCrop
trans_crop = transforms.RandomCrop((20, 20))
trans_compose_2 = transforms.Compose([trans_crop, trans_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()