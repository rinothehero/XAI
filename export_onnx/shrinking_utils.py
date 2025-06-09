import torch
import torch.nn as nn

def extract_nonzero_rows(layer):
    weight = layer.weight.detach().cpu()
    row_norms = torch.norm(weight, p=1, dim=1)
    keep_rows = (row_norms != 0).nonzero(as_tuple=True)[0]
    if len(keep_rows) == 0:
        print("⚠️ 모든 row가 pruning됨. 최소 1개 유지합니다.")
        keep_rows = torch.tensor([0]) # Keep at least one row to prevent layer collapse
    return keep_rows

def shrink_linear_layer(old_layer):
    keep_rows = extract_nonzero_rows(old_layer)
    in_features = old_layer.in_features
    out_features = len(keep_rows)

    # Ensure out_features is not zero
    if out_features == 0:
        # This case should ideally be handled by extract_nonzero_rows ensuring at least one row is kept
        # If it still happens, it means the layer was entirely zero and extract_nonzero_rows logic might need review
        # For now, let's log and potentially create a minimal layer to avoid crashing
        print(f"Error: shrink_linear_layer for a layer of size {in_features} resulted in 0 output features. Check pruning.")
        # Fallback: create a layer that maps to 1 output feature to prevent errors downstream
        # This is a recovery mechanism; the root cause (over-pruning) should be addressed.
        # new_layer = nn.Linear(in_features, 1, bias=old_layer.bias is not None)
        # new_layer.weight.data.zero_() # Zero out weights of this fallback layer
        # if old_layer.bias is not None:
        #    new_layer.bias.data.zero_()
        # return new_layer, torch.tensor([0]) # Indicating the kept row index (arbitrary for fallback)
        # Re-asserting the fix from extract_nonzero_rows should make this rare
        raise ValueError(f"Shrinking layer {old_layer} resulted in zero output features despite safeguards.")


    new_layer = nn.Linear(in_features, out_features, bias=old_layer.bias is not None)
    new_layer.weight.data = old_layer.weight.data[keep_rows].clone()
    if old_layer.bias is not None:
        new_layer.bias.data = old_layer.bias.data[keep_rows].clone()
    return new_layer, keep_rows

def apply_structural_shrink(model):
    # This function assumes a BERT-like structure.
    # It might need adjustment if the model architecture varies significantly.
    if not hasattr(model, 'bert') or not hasattr(model.bert, 'encoder'):
        print("Warning: Model does not have the expected BERT structure (model.bert.encoder). Skipping structural shrink.")
        return model

    for i, layer in enumerate(model.bert.encoder.layer):
        if not (hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense') and
                hasattr(layer, 'output') and hasattr(layer.output, 'dense')):
            print(f"Warning: Layer {i} does not have expected intermediate.dense or output.dense. Skipping.")
            continue

        # Shrink intermediate.dense
        old_inter = layer.intermediate.dense
        if not isinstance(old_inter, nn.Linear):
            print(f"Warning: Layer {i} intermediate.dense is not nn.Linear. Skipping.")
            continue
        new_inter, keep_rows_inter = shrink_linear_layer(old_inter)
        layer.intermediate.dense = new_inter

        # Shrink output.dense (adjusting input dimension based on kept rows from intermediate)
        old_output = layer.output.dense
        if not isinstance(old_output, nn.Linear):
            print(f"Warning: Layer {i} output.dense is not nn.Linear. Skipping.")
            # Restore intermediate layer if output can't be processed, to maintain consistency
            layer.intermediate.dense = old_inter
            continue

        # Check if old_output.weight.data has enough columns for the keep_rows_inter indexing
        if old_output.weight.data.shape[1] < len(keep_rows_inter):
             print(f"Warning: Layer {i} output.dense weight columns ({old_output.weight.data.shape[1]}) < keep_rows_inter ({len(keep_rows_inter)}). This indicates an issue with pruning consistency or layer dimensions. Skipping shrink for this layer.")
             # Restore intermediate layer if output can't be processed
             layer.intermediate.dense = old_inter
             continue


        new_output_in_features = len(keep_rows_inter)
        if new_output_in_features == 0:
            # This should be prevented by shrink_linear_layer safeguards ensuring keep_rows_inter is not empty.
            print(f"Error: Intermediate layer {i} shrinking resulted in zero features for output layer. Skipping shrink for this BERT layer.")
            layer.intermediate.dense = old_inter # Restore previous intermediate layer
            continue

        new_output = nn.Linear(new_output_in_features, old_output.out_features, bias=old_output.bias is not None)

        # Ensure keep_rows_inter indices are valid for old_output.weight.data column dimension
        # This is a sanity check, as keep_rows_inter comes from the output features of the *previous* layer (old_inter),
        # which should match the input features of old_output.
        if not torch.all(keep_rows_inter < old_output.weight.data.shape[1]):
            print(f"Critical Error: Inconsistent dimensions for layer {i}. Indices for shrinking output layer are out of bounds for the original weight matrix. Skipping shrink for this BERT layer.")
            # Restore intermediate layer
            layer.intermediate.dense = old_inter
            continue

        new_output.weight.data = old_output.weight.data[:, keep_rows_inter].clone()
        if old_output.bias is not None:
            new_output.bias.data = old_output.bias.data.clone()
        layer.output.dense = new_output

    # Shrink the pooler.dense layer if it exists
    if hasattr(model, 'bert') and hasattr(model.bert, 'pooler') and hasattr(model.bert.pooler, 'dense') and isinstance(model.bert.pooler.dense, nn.Linear):
        old_pooler_dense = model.bert.pooler.dense
        # The input to pooler.dense is the output of the last transformer layer's first token (usually CLS token embedding)
        # Its input dimension should match the hidden size of the transformer, which is old_pooler_dense.in_features.
        # If the transformer layers themselves were shrunk in their output dimension (e.g. layer.output.dense.out_features),
        # this pooler might not need shrinking unless its own weights were pruned.
        # For this generic shrink utility, we assume its weights might have been pruned directly.
        # If its inputs were changed by a previous shrinking step affecting all bert output hidden states,
        # then this logic might need to be more sophisticated or coordinated.
        # However, apply_structural_shrink primarily targets MLP blocks within encoder layers.
        # Let's assume for now we only shrink it based on its own pruned rows if any.

        # Note: Shrinking pooler.dense like this is only valid if its *output features* were pruned.
        # If its *input features* changed due to BERT encoder layers shrinking, this is not enough.
        # The current `apply_structural_shrink` for BERT layers changes `layer.output.dense` which affects
        # the hidden states fed to subsequent layers AND to the pooler.
        # The `old_output.out_features` of the last BERT layer.output.dense IS the `in_features` for the pooler.dense.
        # So, if `old_output.out_features` was changed, the pooler's `in_features` must be updated.
        # The current shrink logic does not modify `out_features` of `layer.output.dense`, it modifies `in_features` based on `intermediate.dense`.
        # Therefore, pooler.dense's `in_features` dimension remains consistent with BERT's hidden size.
        # We only shrink its output dimension based on its own pruned rows.

        new_pooler_dense, _ = shrink_linear_layer(old_pooler_dense) # We don't need keep_rows for the pooler's output
        model.bert.pooler.dense = new_pooler_dense
        print("Shrank model.bert.pooler.dense layer.")

    return model
