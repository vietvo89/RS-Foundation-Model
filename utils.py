from huggingface_hub import hf_hub_download
import torch, open_clip
import pandas as pd
from os.path import join, exists
from copy import deepcopy
from pathlib import Path
import json

# Path and dictionary
labels_json_pth = '/kaggle/input/train_labels/xView_train.geojson'
labels_csv_pth = '/kaggle/working/labels.csv'
class_map_pth = '/kaggle/working/xView_class_map.csv'
image_fldr = Path('/kaggle/input/train_images/train_images')

old_dict = {
        11:'Fixed-wing Aircraft', 12:'Small Aircraft', 13:'Passenger/Cargo Plane', 15:'Helicopter',
        17:'Passenger Vehicle', 18:'Small Car', 19:'Bus', 20:'Pickup Truck', 21:'Utility Truck',
        23:'Truck', 24:'Cargo Truck', 25:'Truck Tractor w/ Box Trailer', 26:'Truck Tractor',27:'Trailer',
        28:'Truck Tractor w/ Flatbed Trailer', 29:'Truck Tractor w/ Liquid Tank', 32:'Crane Truck',
        33:'Railway Vehicle', 34:'Passenger Car', 35:'Cargo/Container Car', 36:'Flat Car', 37:'Tank car',
        38:'Locomotive', 40:'Maritime Vessel', 41:'Motorboat', 42:'Sailboat', 44:'Tugboat', 45:'Barge',
        47:'Fishing Vessel', 49:'Ferry', 50:'Yacht', 51:'Container Ship', 52:'Oil Tanker',
        53:'Engineering Vehicle', 54:'Tower crane', 55:'Container Crane', 56:'Reach Stacker',
        57:'Straddle Carrier', 59:'Mobile Crane', 60:'Dump Truck', 61:'Haul Truck', 62:'Scraper/Tractor',
        63:'Front loader/Bulldozer', 64:'Excavator', 65:'Cement Mixer', 66:'Ground Grader', 71:'Hut/Tent',
        72:'Shed', 73:'Building', 74:'Aircraft Hangar', 76:'Damaged Building', 77:'Facility', 79:'Construction Site',
        83:'Vehicle Lot', 84:'Helipad', 86:'Storage Tank', 89:'Shipping container lot', 91:'Shipping Container',
        93:'Pylon', 94:'Tower'}

superclass_dict = {'Fixed-wing Aircraft': ['Fixed-wing Aircraft','Small Aircraft','Passenger/Cargo Plane'],
                   'Helicopter':['Helicopter'], 
                   'Passenger Vehicle': ['Passenger Vehicle','Small Car', 'Bus'],
                   'Truck':['Truck','Pickup Truck', 'Utility Truck', 'Cargo Truck', 'Truck Tractor w/ Box Trailer', 'Truck Tractor', 'Trailer',
                            'Truck Tractor w/ Flatbed Trailer', 'Truck Tractor w/ Liquid Tank'],
                   'Railway Vehicle': ['Railway Vehicle','Passenger Car', 'Cargo/Container Car', 'Flat Car', 'Tank car', 'Locomotive'], 
                   'Maritime Vessel':['Maritime Vessel','Motorboat', 'Sailboat', 'Tugboat', 'Barge', 'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker'],
                   'Engineering vessel':['Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 'Straddle Carrier', 'Mobile Crane', 'Dump Truck', 'Haul Truck', 'Scraper/Tractor', 'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader', 'Crane Truck'],
                   'Building':['Building','Hut/Tent', 'Shed', 'Aircraft Hangar', 'Damaged Building', 'Facility'],
                   'Construction Site':['Construction Site'], 
                   'Vehicle Lot':['Vehicle Lot'], 
                   'Helipad':['Helipad'], 
                   'Storage Tank':['Storage Tank'], 
                   'Shipping container lot':['Shipping container lot'], 
                   'Shipping Container':['Shipping Container'], 
                   'Pylon':['Pylon'], 
                   'Tower':['Tower']}

# RemoteCLIP and GeoRSCLIP
def load_model(model_name, device, checkpoint_path=None):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    checkpoint = torch.load(checkpoint_path,weights_only=True, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.eval().to(device)
    
    return model, preprocess, tokenizer

# SkyCLIP
def load_model_skyclip(model_name, device, checkpoint_path=None, src_path_to_skyscript=None):
    import sys
    sys.path.append(src_path_to_skyscript)
    from src.open_clip.factory import create_model_and_transforms, get_tokenizer

    model, _, preprocess = create_model_and_transforms(
                model_name,
                checkpoint_path,
                precision='amp',
                device=device,
                output_dict=True,
                force_quick_gelu=True,
            )

    tokenizer = get_tokenizer(model_name)
    model.eval().to(device)
    
    return model, preprocess, tokenizer

def map_class(old_class,old_dict=None, new_dict=None):
    class_string = old_dict[old_class]
    return new_dict[class_string]

def load_xview_dataset():

    # 1. Load data
    with open(labels_json_pth, 'r') as infile:
        data = json.load(infile)
    feature_list = data['features'] #A list of dicts

    # 2. Create a dataframe
    COLUMNS = ['IMAGE_ID', 'TYPE_ID', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'LONG', 'LAT']
    data = []
    for feature in feature_list:
        properties = feature['properties'] #A dict
        img_id = properties['image_id']  #'389.tif'
        type_id = properties['type_id'] #23
        bbox = properties['bounds_imcoords'].split(",")  #'1917,38,1958,64'
        geometry = feature['geometry']
        coords = geometry['coordinates'][0] #for some reason it's a list of lists
        long = coords[0][0] / 2  + coords[2][0] / 2
        lat = coords[0][1] / 2  + coords[1][1] / 2
        one_row = [img_id, type_id, bbox[0], bbox[1], bbox[2], bbox[3], long, lat]
        data.append(one_row)

    df = pd.DataFrame(data, columns = COLUMNS)
    df[['XMIN', 'YMIN', 'XMAX', 'YMAX']] = df[['XMIN', 'YMIN', 'XMAX', 'YMAX']].apply(pd.to_numeric)
    
    # 3. Filter out the classes
    old_keys = sorted(list(old_dict.keys()))  #not strictly necessary, but makes the above list robust to accidental re-ordering
    df = df[(df.TYPE_ID != 75) & (df.TYPE_ID != 82)]

    new_dict = {old_dict[x]:y for y, x in enumerate(old_keys)}
    pd.DataFrame(new_dict.keys(), index = new_dict.values()).to_csv(class_map_pth, header=False)

    new_classes = df.apply(lambda row: map_class(row['TYPE_ID'],old_dict, new_dict), axis=1)

    # def map_class(old_class):
    #     class_string = old_dict[old_class]
    #     return new_dict[class_string]
    # new_classes = df.apply(lambda row: map_class(row['TYPE_ID']), axis=1)

    df['TYPE_ID'] = new_classes
    df = df[df.IMAGE_ID != '1395.tif']

    # 4. create id_label_dict
    id_label_dict = {idx:label for label,idx in new_dict.items()}

    # 5. create superclass_mapped
    superclass_dict_reversed={}
    for k,v in superclass_dict.items():
        name = ''
        for n in v:
            if len(name)==0:
                name += n    
            else:
                name += ', '+n
        superclass_dict_reversed[name] = k

    superclass_mapped={}
    for k,v in new_dict.items():
        for k1,v1 in superclass_dict_reversed.items():
            if k in k1:
                superclass_mapped[k]=v1            

    return df, id_label_dict, superclass_mapped

def get_boxes(in_df, class_list=[]):
    if class_list:
        # keep only rows where TYPE_ID IS in the class list
        in_df = in_df[in_df['TYPE_ID'].isin(class_list)]

    unique_images = in_df.IMAGE_ID.unique().tolist()
    boxs = {}
    for image in unique_images:
        mask = in_df['IMAGE_ID'] == image
        masked = in_df[mask][['TYPE_ID', 'XMIN', 'YMIN', 'XMAX', 'YMAX']]
        boxs[image] = masked.values.tolist()
    return boxs

def get_features(model, tokenizer, new_dict):
    context_length = 77
    with torch.no_grad():
        text_descriptions = [f"A photo of a {label}" for label in new_dict]
        inputs = tokenizer(text_descriptions,context_length=context_length)
        text_features = model.encode_text(inputs.cuda())    
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features