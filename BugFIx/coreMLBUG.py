import torch
import coremltools as ct


# Define a simple PyTorch model that uses grid sampling
class PytorchGridSample(torch.nn.Module):
    def forward(self, input, grid):
        return torch.nn.functional.grid_sample(input, grid, align_corners=False)


def convert_to_coreml(model, inputs, is_float16=True):
    traced_model = torch.jit.trace(
        model, example_inputs=inputs, strict=False)

    coreml_model = ct.converters.convert(traced_model,
                                         inputs=[ct.TensorType(shape=inputs[0].shape),
                                                 ct.TensorType(shape=inputs[1].shape)],
                                         compute_precision=ct.precision.FLOAT16 if is_float16 else ct.precision.FLOAT32)
    return coreml_model


def compare_grid_samples_after_coreml_conversion(pt_model, inputs, is_float16):
    """
    Compare the grid sample output before and after conversion to coreML
    """
    pt_out = pt_model(*inputs)
    coreml_pt_model = convert_to_coreml(pt_model, inputs, is_float16)
    input_names_coreml_pt = [i for i in
                             coreml_pt_model.input_description]
    input_data = {name: val.detach().numpy() for name, val in zip(input_names_coreml_pt, inputs)}

    coreml_pt_out = torch.as_tensor(list(coreml_pt_model.predict(input_data).values())[0])
    diff_pt_coreml = torch.norm(coreml_pt_out - pt_out)
    return diff_pt_coreml


if __name__ == "__main__":
    input_tensor = torch.load("feat_tensor.pt").to(torch.float32)
    grid = torch.load("grid_tensor.pt").to(torch.float32)
    inputs = [input_tensor, grid]
    pt_model = PytorchGridSample()

    diff_pt_coreml_fp16 = compare_grid_samples_after_coreml_conversion(pt_model, [*inputs], is_float16=True)
    diff_pt_coreml_fp32 = compare_grid_samples_after_coreml_conversion(pt_model, [*inputs],  is_float16=False)

    print(
        f"Difference between pytorch's grid sample before and after conversion: Note: Pytorch is fp32 and coreML is fp16 : {diff_pt_coreml_fp16}")

    print(
        f"Difference between pytorch's grid sample before and after conversion: Note: Pytorch is fp32 and coreML is fp32 : {diff_pt_coreml_fp32}")

    print(f"Relative change in the difference: {(diff_pt_coreml_fp16 - diff_pt_coreml_fp32) / diff_pt_coreml_fp32}")