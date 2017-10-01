import pandas as pd
import yaml

def reformat(y, save_dir):

    # define mapping
    namemap = {'Green': 'green', 'GreenLeft': 'green', 'GreenRight': 'green',
    'GreenStraight': 'green', 'GreenStraightLeft': 'green', 'GreenStraightRight': 'green',
    'Red': 'red', 'RedLeft': 'red', 'RedRight': 'red', 'RedStraight': 'red',
    'RedStraightLeft': 'red', 'RedStraightRight': 'red', 'Yellow': 'yellow'}

    # definte format
    annotation = """
{{
  "filename" : "{0}",
  "folder" : "{1}",
  "image_w_h" : [
    1280,
    720
  ],
  "objects" : [
  {2}
  ]
}}
"""
    object_chunk = """
    {{
      "label" : "traffic_light-{0}",
      "x_y_w_h" : [
        {1},
        {2},
        {3},
        {4}
      ]
    }}
"""

    # Read the raw yaml
    with open(y, 'r') as f:
        df = pd.io.json.json_normalize(yaml.load(f))

    # restructure
    for index, row in df.iterrows():
        obs = []
        for box in row['boxes']:
            if box['label'] != 'off':
                label = namemap[box['label']]
                x = box['x_min']
                y = box['y_min']
                w = box['x_max']-x
                h = box['y_max']-y
                obs.append(object_chunk.format(label, x, y, w, h))
        filename = row["path"].split('/')[-1]
        location = row["path"][:len(filename)]
        obs = ', '.join(obs)
        annote = annotation.format(filename, location, obs)

        # write to file
        with open(save_dir + "/" +
            filename.split('.')[0] + "_bosch.json", "w") as saveloc:
            saveloc.write(annote)

def main():
    
    yamllist = ['/home/kevin/Downloads/dataset_additional_rgb/additional_train.yaml']
    save_dir = '/home/kevin/Downloads/dataset_additional_rgb/annotations'
    for y in yamllist:
        reformat(y, save_dir)


if __name__ == '__main__':
    main()