import torch
import coremltools as ct

# Pytorch Grid Model
class PytorchGridSample(torch.nn.Module):
    def forward(self, input, grid):
        return torch.nn.functional.grid_sample(input, grid, align_corners=True)

def generate_random_tensors(shape):
    return torch.randn(shape)

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
        print("After conversion to CoreML")

        # Generate random tensors as inputs
        random_input_tensor = generate_random_tensors(random_tensor_shape)
        random_grid_tensor = generate_random_tensors((random_tensor_shape[0], random_tensor_shape[2], random_tensor_shape[3], 2))
        random_tensors = [random_input_tensor, random_grid_tensor]

        pt_model = PytorchGridSample()

        diff_pt_coreml_fp16 = compare_grid_samples_after_coreml_conversion(pt_model, random_tensors, is_float16=True)
        diff_pt_coreml_fp32 = compare_grid_samples_after_coreml_conversion(pt_model, random_tensors, is_float16=False)

        coreml_pt_model = convert_to_coreml(pt_model, random_inputs, is_float16)

        if coreml_pt_model is not None:
            # debugging
            print(f"Random Input tensor: {random_input}")
            print(f"Random Grid tensor: {random_grid}")
            print(f"PyTorch output: {pt_out}")

            input_names_coreml_pt = [i for i in coreml_pt_model.input_description]
            input_data = {name: val.detach().numpy() for name, val in zip(input_names_coreml_pt, random_inputs)}

            coreml_pt_out = torch.as_tensor(list(coreml_pt_model.predict(input_data).values())[0])
            print(f"CoreML output: {coreml_pt_out}")

            diff_pt_coreml = torch.norm(coreml_pt_out - pt_out)
            return diff_pt_coreml
        else:
            return None
    except Exception as e:
        print(f"Error during comparison: {e}")
        return None

if __name__ == "__main__":
    try:
        # Specify the shape of the random tensors
        random_tensor_shape = (32, 3, 224, 224)# Replace with actual shape
          

        # Generate random tensors
        random_input_tensor = generate_random_tensors(random_tensor_shape)
        random_grid_tensor = generate_random_tensors(random_tensor_shape)
        random_tensors = [random_input_tensor, random_grid_tensor]

        pt_model = PytorchGridSample()

        diff_pt_coreml_fp16 = compare_grid_samples_after_coreml_conversion(pt_model, random_tensors, is_float16=True)
        diff_pt_coreml_fp32 = compare_grid_samples_after_coreml_conversion(pt_model, random_tensors, is_float16=False)

        if diff_pt_coreml_fp16 is not None and diff_pt_coreml_fp32 is not None:
            print(f"Difference between PyTorch's grid sample before and after conversion: Note: PyTorch is fp32 and CoreML is fp16: {diff_pt_coreml_fp16}")
            print(f"Difference between PyTorch's grid sample before and after conversion: Note: PyTorch is fp32 and CoreML is fp32: {diff_pt_coreml_fp32}")
            print(f"Relative change in the difference: {(diff_pt_coreml_fp16 - diff_pt_coreml_fp32) / diff_pt_coreml_fp32}")
    except Exception as e:
        print(f"Unexpected error: {e}")
