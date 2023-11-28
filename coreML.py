import torch
import coremltools as ct

# Pytorch Grid Model
class PytorchGridSample(torch.nn.Module):
    def forward(self, input, grid):
        return torch.nn.functional.grid_sample(input, grid, align_corners=True)

def convert_to_coreml(model, inputs, is_float16=True):
    try:
        traced_model = torch.jit.trace(model, example_inputs=inputs, strict=False)
        
        # Precision 
        precision = ct.precision.FLOAT16 if is_float16 else ct.precision.FLOAT32
        coreml_model = ct.converters.convert(
            traced_model,
            inputs=[ct.TensorType(shape=inputs[0].shape),
                    ct.TensorType(shape=inputs[1].shape)],
            compute_precision=precision
        )
        
        return coreml_model
    except Exception as e:
        print(f"Error during CoreML conversion: {e}")
        return None

def compare_grid_samples_after_coreml_conversion(pt_model, inputs, is_float16):
    try:
        print("Before conversion to CoreML")
        pt_out = pt_model(*inputs)
        print("PyTorch output:")
        print(pt_out)

        print("After conversion to CoreML")
        coreml_pt_model = convert_to_coreml(pt_model, inputs, is_float16)

        device = torch.device("cpu")
        input_tensor = inputs[0].to(device)
        grid = inputs[1].to(device)
        pt_model.to(device)

        if coreml_pt_model is not None:
            # debugging
            print(f"Input tensor: {input_tensor}")
            print(f"Grid tensor: {grid}")

            input_names_coreml_pt = [i for i in coreml_pt_model.input_description]
            input_data = {name: val.detach().numpy() for name, val in zip(input_names_coreml_pt, inputs)}

            print("CoreML output before converting to PyTorch tensor:")
            coreml_pt_out_raw = coreml_pt_model.predict(input_data).values()
            print(coreml_pt_out_raw)

            coreml_pt_out = torch.as_tensor(list(coreml_pt_out_raw)[0])
            print("CoreML output after converting to PyTorch tensor:")
            print(coreml_pt_out)

            diff_pt_coreml = torch.norm(coreml_pt_out - pt_out)
            print(f"Difference between PyTorch's grid sample before and after conversion: {diff_pt_coreml}")
            
            return diff_pt_coreml
        else:
            return None
    except Exception as e:
        print(f"Error during comparison: {e}")
        return None

if __name__ == "__main__":
    try:
        # Specify the paths to the extracted tensors
        grid_tensor_path = "/Users/surajmahapatra/Desktop/coreML Bug/grid_tensor.pt"
        feature_tensor_path = "/Users/surajmahapatra/Desktop/coreML Bug/input_tensor.pt"

        # Load the extracted tensors
        input_tensor = torch.load(feature_tensor_path).to(torch.float32)
        grid = torch.load(grid_tensor_path).to(torch.float32)
        inputs = [input_tensor, grid]
        pt_model = PytorchGridSample()

        diff_pt_coreml_fp16 = compare_grid_samples_after_coreml_conversion(pt_model, [*inputs], is_float16=True)
        diff_pt_coreml_fp32 = compare_grid_samples_after_coreml_conversion(pt_model, [*inputs], is_float16=False)

        if diff_pt_coreml_fp16 is not None and diff_pt_coreml_fp32 is not None:
            print(f"Difference between PyTorch's grid sample before and after conversion: Note: PyTorch is fp32 and CoreML is fp16: {diff_pt_coreml_fp16}")
            print(f"Difference between PyTorch's grid sample before and after conversion: Note: PyTorch is fp32 and CoreML is fp32: {diff_pt_coreml_fp32}")
            print(f"Relative change in the difference: {(diff_pt_coreml_fp16 - diff_pt_coreml_fp32) / diff_pt_coreml_fp32}")
    except Exception as e:
        print(f"Unexpected error: {e}")
