from .instruct_ir import LoadInstructIRModel, InstructIRProcess
from .install import check_and_install


check_and_install("Pillow","PIL")
check_and_install("huggingface_hub")
check_and_install("transformers")
check_and_install("PyYAML","yaml")
check_and_install("sentence-transformers","sentence_transformers")


NODE_CLASS_MAPPINGS = {
    "InstructIRProcess": InstructIRProcess,
    "LoadInstructIRModel": LoadInstructIRModel,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "InstructIRProcess": "InstructIR Process Image",
    "LoadInstructIRModel": "Loader InstructIR Model",
}