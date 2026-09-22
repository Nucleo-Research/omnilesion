"""3D RetinaNet nodule detector built from MONAI library components (Apache-2.0), trained from scratch.

ResNet-50 backbone (3D, stride-2 stem), feature pyramid over three levels, three cubic anchors of 4/6/8 mm per
location, ATSS matching and hard-negative sampling during training. Input: a lung-cropped volume at 1 mm isotropic
spacing windowed to [-1024, 300] HU and scaled to [0, 1].
"""

from __future__ import annotations

ANCHORS = ((4, 4, 4), (6, 6, 6), (8, 8, 8))
PATCH = [128, 128, 128]


def build_detector(device, patch=PATCH, training: bool = True, sw_batch_size: int = 1):
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets import resnet

    anchor_generator = AnchorGeneratorWithAnchorShape(feature_map_scales=[1, 2, 4], base_anchor_shapes=list(ANCHORS))
    backbone = resnet.resnet50(spatial_dims=3, n_input_channels=1, conv1_t_stride=[2, 2, 2], conv1_t_size=[7, 7, 7])
    feature_extractor = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
    network = RetinaNet(spatial_dims=3, num_classes=1, num_anchors=3, feature_extractor=feature_extractor,
                        size_divisible=[16, 16, 16], use_list_output=False)
    detector = RetinaNetDetector(network=network, anchor_generator=anchor_generator, debug=False).to(device)
    if training:
        detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
        detector.set_hard_negative_sampler(batch_size_per_image=64, positive_fraction=0.3, pool_size=20, min_neg=16)
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(score_thresh=0.02, topk_candidates_per_level=1000, nms_thresh=0.22,
                                         detections_per_img=300)
    detector.set_sliding_window_inferer(roi_size=list(patch), overlap=0.25, sw_batch_size=sw_batch_size,
                                        mode="constant", device="cpu")
    return detector
