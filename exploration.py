from utils import *
from visualization import *

def show_predict_image_patching(model, preprocess, text_features, id_label_dict, img_nm=None, patch_size=224,stride=224, zoom=1):
    
    img_pth = str(image_fldr / img_nm)
    im = cv2.imread(img_pth)    # tried using the flag cv2.IMREAD_ANYDEPTH flag, because images are 24 bit, but that removed the color channels.
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    dh, dw, _ = im.shape #high for row and width for col
    # 1. full image classification
    image = Image.fromarray(im).convert("RGB")
    inputs = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(inputs.cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
            
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    _, top_labels = text_probs.cpu().topk(5, dim=-1)
    
    # 2. create a bounding box with the data & draw it
    no_cols = (dw-patch_size-1)//stride+1
    no_rows = (dh-patch_size-1)//stride+1 #n_grids//no_cols
    _, axs = plt.subplots(no_rows, no_cols, figsize=(3*no_rows, 3*no_cols))
    i=0
    for i in range(no_rows):
        for j in range(no_cols):
            y_max, x_min  = int(i*stride), int(j*stride)
            img_box=im[y_max: y_max+patch_size,x_min: x_min + patch_size,:]
            image = Image.fromarray(img_box).convert("RGB")
            inputs = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                image_features = model.encode_image(inputs.cuda())
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, top_labels = text_probs.cpu().topk(5, dim=-1)
            
            img_box = cv2.resize(img_box, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    
            # Show image of boxes
            axs[i][j].set_title(f"{i}/{j}-{id_label_dict[top_labels[0,0].item()]}({top_labels[0,0].item()})", fontsize = 12)
            axs[i][j].imshow(img_box)
            axs[i][j].set_axis_off()

    plt.tight_layout()
    plt.show()