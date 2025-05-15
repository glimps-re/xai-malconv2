import io

import lief
import numpy as np
import pefile
import torch
from elftools.elf.elffile import ELFFile

from malconv.inference.MalConvGCT_nocat_Inf import Extracted_MLP, MalConvGCT


class MalConv2ModelError(Exception):
    """MalConv2 custom exception."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def to_dict(self):
        """Return dict form"""
        return {"message": self.message}


def get_exec_sections_elf(file_path: str):
    with open(file_path, "rb") as f:
        file_data = f.read()
        # Create a file-like object from the buffered ELF data
        elf_file_from_mem = io.BytesIO(file_data)
        elf_file = ELFFile(elf_file_from_mem)
        exec_sections = []
        raw_bytes = []
        for section in elf_file.iter_sections():
            if (section["sh_flags"] & 0x4) == 0x4:
                try:
                    name = section.Name.decode().rstrip("\x00")
                    name = name[:10]
                except:
                    name = "ERROR"
                exec_sections.append(
                    (name, section["sh_offset"], section["sh_size"], section["sh_addr"], section["sh_size"])
                )
                f.seek(section["sh_offset"])
                raw_bytes += f.read(section["sh_size"])
        return raw_bytes, exec_sections


def get_exec_sections_pe(file_path: str):
    with open(file_path, "rb") as f:
        file_data = f.read()
        pe = pefile.PE(data=file_data)
        exec_sections = []
        raw_bytes = []
        for section in pe.sections:
            if (
                pefile.SECTION_CHARACTERISTICS["IMAGE_SCN_MEM_EXECUTE"] & section.Characteristics
                == pefile.SECTION_CHARACTERISTICS["IMAGE_SCN_MEM_EXECUTE"]
            ):
                try:
                    name = section.Name.decode().rstrip("\x00")
                    name = name[:10]
                except:
                    name = "ERROR"
                exec_sections.append(
                    (
                        name,
                        section.PointerToRawData,
                        section.SizeOfRawData,
                        section.VirtualAddress,
                        section.Misc_VirtualSize,
                    )
                )
                f.seek(section.PointerToRawData)
                raw_bytes += f.read(section.SizeOfRawData)
        return raw_bytes, exec_sections


class MalConv2Model:
    def __init__(self, model_name: str = "epoch_310.model"):
        super().__init__()
        self.model = MalConvGCT(
            channels=256,
            window_size=256,
            stride=64,
        )
        x = torch.load(model_name, map_location=torch.device("cpu"))
        self.model.load_state_dict(x["model_state_dict"], strict=False)
        self.model.eval()

        # getting out just the MLP
        self.mlp_model = Extracted_MLP()
        processed_dict = {}
        for k in ["fc_1.weight", "fc_1.bias", "fc_2.weight", "fc_2.bias"]:
            processed_dict[k] = self.model.state_dict()[k]
        self.mlp_model.load_state_dict(processed_dict)
        self.mlp_model.eval()

    def predict(self, x):
        try:
            with torch.no_grad():
                post_conv, _ = self.model(x)
                output = self.mlp_model(post_conv)
            prediction = output.detach().numpy()[0]
        except Exception:
            raise MalConv2ModelError("Error while computing file prediction") from None
        return prediction

    def __call__(self, filepath: str, exp_method: callable, logger=None):
        raw_bytes = []

        if lief.is_pe(filepath):
            raw_bytes, exec_sections = get_exec_sections_pe(filepath)
        elif lief.is_elf(filepath):
            raw_bytes, exec_sections = get_exec_sections_elf(filepath)
        else:
            raise MalConv2ModelError("Neither a PE or an ELF executable file")

        try:
            x_np = np.frombuffer(bytearray(raw_bytes), dtype=np.uint8).astype(np.int16) + 1
            x_np = x_np.reshape((1, -1))
            if x_np.shape[1] < 2000:
                x_np = np.pad(
                    x_np, ((0, 0), (0, 2000 - x_np.shape[1])), "constant"
                )  # pad to have a min. length equal to 2000

            x = torch.Tensor(x_np).type(torch.int16)

            with torch.no_grad():
                post_conv, indices = self.model(x)
                output = self.mlp_model(post_conv)

            sample = post_conv.detach().numpy()
            attr = exp_method(sample)
            if len(attr.shape) > 0:
                attr = attr[0]
            _dict = {}

            heatmap = np.zeros((max(x_np.shape[1], 2000),), dtype=np.float32)
            for i in range(256):
                heatmap[indices[0][i]] += attr[i]
                if indices[0][i] in _dict:
                    _dict[indices[0][i]] += attr[i]
                else:
                    _dict[indices[0][i]] = attr[i]

            # change the scale of the heatmap to fit the prediction of the model
            max_ = heatmap.max()
            min_ = heatmap.min() + 1e-12
            sum_p = heatmap[heatmap > 0].sum()
            sum_n = heatmap[heatmap < 0].sum()
            prediction = output.detach().numpy()[0]

        except Exception:
            raise MalConv2ModelError("Error while computing file prediction") from None

        try:
            heatmap[heatmap > 0] *= abs(prediction) / max_
            heatmap[heatmap < 0] *= abs(1.0 - prediction) / abs(min_)

            # change also the functions scores to fit the prediction of the model
            _dict = {k: (v / sum_p) if v >= 0.0 else (v / abs(sum_n)) for k, v in _dict.items()}

            filtered_dict = {k: v for k, v in _dict.items() if v > 0.0}
            sorted_dict = dict(sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True))

            # change the dict key to be the file offset
            functions_scores_by_offset = {}
            for key, value in sorted_dict.items():
                # compute the offset in the section
                prev_size = 0
                for _sec_name, _sec_offset, sec_size, _, _ in exec_sections:
                    if key < prev_size + sec_size:
                        # the section is the one were the function is found
                        break
                    prev_size += sec_size
                functions_scores_by_offset[hex(key - prev_size + _sec_offset)] = (
                    value  # and put keys in string format to be json exportable with json.dumps
                )

        except Exception:
            raise MalConv2ModelError("Error during function offset recuperation") from None

        return prediction, functions_scores_by_offset
