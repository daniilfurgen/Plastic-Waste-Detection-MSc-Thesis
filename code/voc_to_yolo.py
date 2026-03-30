import xml.etree.ElementTree as ET
import os

# Define your classes here (must match your data.yaml order)
classes = ["bottle"]


def convert(
    size: tuple[int, int], box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def find_text(element: ET.Element, tag: str) -> str:
    """Finds a tag and returns text, or crashes with a helpful error."""
    found = element.find(tag)
    if found is None or found.text is None:
        raise ValueError(f"Missing required XML tag: {tag}")
    return found.text


def convert_annotation(xml_path: str, output_path: str) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if (size := root.find("size")) is None:
        raise ValueError("Missing required XML tag: size")

    w = int(find_text(size, "width"))
    h = int(find_text(size, "height"))

    with open(output_path, "w") as out_file:
        for obj in root.iter("object"):
            cls = find_text(obj, "name")
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            if (xmlbox := obj.find("bndbox")) is None:
                continue
            b = (
                float(find_text(xmlbox, "xmin")),
                float(find_text(xmlbox, "xmax")),
                float(find_text(xmlbox, "ymin")),
                float(find_text(xmlbox, "ymax")),
            )
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + "\n")


if __name__ == "__main__":
    # input_dir = "code/FLoW-img/val/annotations"
    # output_dir = "code/FLoW-img/val/labels"
    input_dir = "code/FLoW-img/train/annotations"
    output_dir = "code/FLoW-img/train/labels"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".xml", ".txt"))
            convert_annotation(xml_path, output_path)
