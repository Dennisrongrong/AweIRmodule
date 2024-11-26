import torch

def vprint(*args, verbose=True):
    """
    打印调试信息的辅助函数，当 verbose 为 True 时启用。
    """
    if verbose:
        print(*args)


def compute_compound_loss(
    criterion_dict: dict,
    raw_network_outputs: torch.Tensor,
    label: torch.Tensor,
    blob_loss_mode=False,
    masked=True,
    verbose=False,
):
    """
    计算复合损失函数，支持多种损失函数的加权组合。

    参数：
        criterion_dict (dict): 损失函数配置字典，每个损失函数包括名称、权重等信息。
        raw_network_outputs (torch.Tensor): 网络的原始输出张量。
        label (torch.Tensor): 真实标签。
        blob_loss_mode (bool): 是否计算 blob（区域）损失。
        masked (bool): 是否在 blob 损失中应用掩码。
        verbose (bool): 是否启用调试打印。

    返回值：
        torch.Tensor: 计算的总损失值。
    """
    losses = []
    for entry in criterion_dict.values():
        name = entry["name"]
        vprint(f"正在计算损失: {name}", verbose=verbose)
        criterion = entry["loss"]
        weight = entry["weight"]
        sigmoid = entry["sigmoid"]

        # 如果 blob_loss_mode 关闭，则计算普通损失
        if not blob_loss_mode:
            vprint("计算普通损失", verbose=verbose)
            processed_output = torch.sigmoid(raw_network_outputs) if sigmoid else raw_network_outputs
            individual_loss = criterion(processed_output, label)
        else:
            # 计算 blob（区域）损失
            vprint("计算 blob 损失", verbose=verbose)
            processed_output = torch.sigmoid(raw_network_outputs) if sigmoid else raw_network_outputs
            if masked:
                individual_loss = compute_blob_loss_multi(
                    criterion, processed_output, label, verbose
                )
            else:
                individual_loss = compute_no_masking_multi(
                    criterion, processed_output, label, verbose
                )

        # 按权重对单个损失加权
        weighted_loss = individual_loss * weight
        losses.append(weighted_loss)

    return sum(losses)


def compute_blob_loss_multi(criterion, network_outputs, multi_label, verbose=False):
    """
    计算每个 batch 元素的 blob（区域）损失。

    参数：
        criterion: 损失函数（如 BCE 或 Dice）。
        network_outputs (torch.Tensor): 网络输出张量。
        multi_label (torch.Tensor): 每个像素的多标签。
        verbose (bool): 是否启用调试打印。

    返回值：
        float: 计算的平均 blob 损失。
    """
    batch_length = multi_label.size(0)
    element_blob_loss = []

    # 遍历 batch 中的每个样本
    for element in range(batch_length):
        element_label = multi_label[element].unsqueeze(0)
        element_output = network_outputs[element].unsqueeze(0)
        unique_labels = torch.unique(element_label)
        label_loss = []

        # 遍历样本中的每个唯一标签
        for ula in unique_labels:
            if ula > 0:  # 忽略背景（假设背景标签为 0）
                label_mask = (element_label != ula).logical_not()
                masked_output = element_output * label_mask.float()
                blob_loss = criterion(masked_output, (element_label == ula).float())
                label_loss.append(blob_loss)

        # 对当前样本的所有标签损失求平均
        if label_loss:
            element_blob_loss.append(sum(label_loss) / len(label_loss))

    # 返回 batch 的平均 blob 损失
    return sum(element_blob_loss) / len(element_blob_loss) if element_blob_loss else 0


def compute_no_masking_multi(criterion, network_outputs, multi_label, verbose=False):
    """
    计算不使用掩码的 blob 损失（用于对比实验）。

    参数：
        criterion: 损失函数（如 BCE 或 Dice）。
        network_outputs (torch.Tensor): 网络输出张量。
        multi_label (torch.Tensor): 每个像素的多标签。
        verbose (bool): 是否启用调试打印。

    返回值：
        float: 计算的平均 blob 损失。
    """
    batch_length = multi_label.size(0)
    element_blob_loss = []

    for element in range(batch_length):
        element_label = multi_label[element].unsqueeze(0)
        element_output = network_outputs[element].unsqueeze(0)
        unique_labels = torch.unique(element_label)
        label_loss = []

        for ula in unique_labels:
            if ula > 0:  # 忽略背景
                blob_loss = criterion(element_output, (element_label == ula).float())
                label_loss.append(blob_loss)

        if label_loss:
            element_blob_loss.append(sum(label_loss) / len(label_loss))

    return sum(element_blob_loss) / len(element_blob_loss) if element_blob_loss else 0


def compute_loss(
    blob_loss_dict: dict,
    criterion_dict: dict,
    blob_criterion_dict: dict,
    raw_network_outputs: torch.Tensor,
    binary_label: torch.Tensor,
    multi_label: torch.Tensor,
    verbose=False,
):
    """
    计算总损失，包括全局主损失（main loss）和局部 blob 损失（blob loss）。

    参数：
        blob_loss_dict (dict): 包含主损失和 blob 损失的权重配置。
        criterion_dict (dict): 用于主损失的损失函数配置字典。
        blob_criterion_dict (dict): 用于 blob 损失的损失函数配置字典。
        raw_network_outputs (torch.Tensor): 网络的原始输出张量。
        binary_label (torch.Tensor): 全局二值标签。
        multi_label (torch.Tensor): 多标签（用于 blob 损失）。
        verbose (bool): 是否启用调试打印。

    返回值：
        tuple: (总损失, 主损失, blob 损失)。
    """
    main_weight = blob_loss_dict.get("main_weight", 0)
    blob_weight = blob_loss_dict.get("blob_weight", 0)

    # 计算主损失
    main_loss = compute_compound_loss(
        criterion_dict=criterion_dict,
        raw_network_outputs=raw_network_outputs,
        label=binary_label,
        blob_loss_mode=False,
        verbose=verbose,
    ) if main_weight > 0 else 0

    # 计算 blob 损失
    blob_loss = compute_compound_loss(
        criterion_dict=blob_criterion_dict,
        raw_network_outputs=raw_network_outputs,
        label=multi_label,
        blob_loss_mode=True,
        verbose=verbose,
    ) if blob_weight > 0 else 0

    # 加权计算总损失
    loss = main_loss * main_weight + blob_loss * blob_weight
    return loss, main_loss, blob_loss
