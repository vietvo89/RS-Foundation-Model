from PIL import Image
import cv2
from utils import *
from tqdm import tqdm


def image_level_test(model, preprocess, text_features, olabel, df):
    boxes = get_boxes(df, [olabel])
    img_list = list(boxes.keys()) 
    n = len(img_list)
    top_1_acc = 0
    top_5_acc = 0
    for idx in tqdm(range(n)):
        img_path = str(image_fldr / img_list[idx])
        im_tif = cv2.imread(img_path)
        im_tif = cv2.cvtColor(im_tif, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_tif).convert("RGB")
        # inputs = preprocess(im).unsqueeze(0)
        inputs = preprocess(im).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(inputs.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
    
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, top_labels = text_probs.cpu().topk(5, dim=-1)
        top_1_pred = 1 if olabel==top_labels[0,0].numpy() else 0
        top_5_pred = 1 if olabel in list(top_labels[0].numpy()) else 0
        top_1_acc += top_1_pred
        top_5_acc += top_5_pred
    
    print(f'top_1_accuracy={top_1_acc/n*100}, top_5_accuracy={top_5_acc/n*100}')
    return top_1_acc/n*100, top_5_acc/n*100

def image_level_superclass_test(model, preprocess, text_features, olabel, df,id_label_dict, superclass_mapped):
    boxes = get_boxes(df, [olabel])
    img_list = list(boxes.keys()) 
    n = len(img_list)
    top_1_acc = 0
    osuperclass = superclass_mapped[id_label_dict[olabel]]
    
    for idx in tqdm(range(n)):
        img_path = str(image_fldr / img_list[idx])
        im_tif = cv2.imread(img_path)
        im_tif = cv2.cvtColor(im_tif, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_tif).convert("RGB")
        inputs = preprocess(im).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(inputs.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
    
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, top_labels = text_probs.cpu().topk(1, dim=-1)
        top_superclass = superclass_mapped[id_label_dict[top_labels[0,0].item()]]
            
        top_1_pred = 1 if osuperclass==top_superclass else 0
        top_1_acc += top_1_pred
    
    print(f'top_1_acc={top_1_acc/n*100}')
    return top_1_acc/n*100

def object_level_test(model, preprocess, text_features, olabel, df):
    boxes = get_boxes(df, [olabel])
    img_list = list(boxes.keys()) 
    n = 0
    top_1_acc = 0
    top_5_acc = 0
    for idx in tqdm(range(len(img_list))):
        img_nm = img_list[idx]
        img_path = str(image_fldr / img_nm)
        im = cv2.imread(img_path)    # tried using the flag cv2.IMREAD_ANYDEPTH flag, because images are 24 bit, but that removed the color channels.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dh, dw, _ = im.shape
        
        selected_df = df[(df['IMAGE_ID']==img_nm)]
        selected_df=selected_df.reset_index()
        n_sub=0
        top_1_sub_acc = 0
        top_5_sub_acc = 0
        for i, row in selected_df.iterrows():
            box_id, x_min, y_min, x_max, y_max = row['TYPE_ID'],row['XMIN'],row['YMIN'],row['XMAX'],row['YMAX']
            if box_id == olabel:                     
                x_min, y_min, x_max, y_max = int(x_min), int(y_max), int(x_max), int(y_min)
                w=(x_max-x_min)#//4
                h=(y_min-y_max)#//4
                x_c=(x_max+x_min)//2
                y_c=(y_max+y_min)//2
                l = max(w,h)
                scale=2 #if w*h*100<dw*dh else 2
                x_min=max(x_c-l*scale,0)
                y_max=max(y_c-l*scale,0)
                x_max=min(x_c+l*scale,dw)
                y_min=min(y_c+l*scale,dh)

                if x_min==0:
                    x_max=x_min+2*l*scale
                elif x_max==dw:
                    x_min=x_max-2*l*scale
        
                if y_max==0:
                    y_min=y_max+2*l*scale
                elif y_min==dh:
                    y_max=y_min-2*l*scale
                    
                if (x_min<x_max)and(y_max<y_min):
                    n_sub+=1
                    n+=1       
                    img_box=im[y_max: y_min,x_min: x_max,:]
                    image = Image.fromarray(img_box).convert("RGB")
                    inputs = preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        image_features = model.encode_image(inputs.cuda())
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    _, top_labels = text_probs.cpu().topk(5, dim=-1)
                    top_1_pred = 1 if box_id==top_labels[0,0].numpy() else 0
                    top_5_pred = 1 if box_id in list(top_labels[0].numpy()) else 0
                    top_1_sub_acc += top_1_pred
                    top_5_sub_acc += top_5_pred
                    top_1_acc += top_1_pred
                    top_5_acc += top_5_pred

    print(f'top_1_accuracy={top_1_acc/n*100}, top_5_accuracy={top_5_acc/n*100}')
    return top_1_acc/n*100, top_5_acc/n*100