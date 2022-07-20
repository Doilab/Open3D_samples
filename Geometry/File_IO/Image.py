import open3d as o3d

print("Testing IO for images ...")
image_data = o3d.data.JuneauImage()
img = o3d.io.read_image(image_data.path)
print(img)
o3d.io.write_image("copy_of_Juneau.jpg", img)