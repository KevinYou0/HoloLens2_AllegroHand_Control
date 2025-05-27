# IKPy imports
from ikpy import chain
from ikpy.urdf.utils import get_urdf_tree

# Generate the pdf
dot, urdf_tree = get_urdf_tree("/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_hand_description_right.urdf",
                               out_image_path="/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/img.png", root_element="base")

allegro_thumb_link = [
    "base", "link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"
]
allegro_index_link = [
    "base", "link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"
]
allegro_middle_link = [
    "base", "link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"
]
allegro_ring_link = [
    "base", "link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"
]

allegro_thumb_joint = [
    "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"
]
allegro_index_joint = [
    "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip"
]
allegro_middle_joint = [
    "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip"
]
allegro_ring_joint = [
    "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip"
]


allegro_thumb_element = [x for pair in zip(allegro_thumb_link, allegro_thumb_joint) for x in pair]
allegro_index_element = [x for pair in zip(allegro_index_link, allegro_index_joint) for x in pair]
allegro_middle_element = [x for pair in zip(allegro_middle_link, allegro_middle_joint) for x in pair]
allegro_ring_element = [x for pair in zip(allegro_ring_link, allegro_ring_joint) for x in pair]
# Remove the gripper, it's weird
# baxter_left_arm_elements = [x for pair in zip(baxter_left_arm_links, baxter_left_arm_joints) for x in pair][:-3]

allegro_thumb_chain = chain.Chain.from_urdf_file(
    "/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_hand_description_right.urdf",
    base_elements=allegro_thumb_element,
    active_links_mask=[False] + 4 * [True] + [False],
    symbolic=False,
    name="allegro_thumb",
)
allegro_thumb_chain.to_json_file(force=True)

allegro_index_chain = chain.Chain.from_urdf_file(
    "/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_hand_description_right.urdf",
    base_elements=allegro_index_element,
    active_links_mask=[False] + 4 * [True] + [False],
    symbolic=False,
    name="allegro_index",
)
allegro_index_chain.to_json_file(force=True)

allegro_middle_chain = chain.Chain.from_urdf_file(
    "/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_hand_description_right.urdf",
    base_elements=allegro_middle_element,
    active_links_mask=[False] + 4 * [True] + [False],
    symbolic=False,
    name="allegro_middle",
)
allegro_middle_chain.to_json_file(force=True)

allegro_ring_chain = chain.Chain.from_urdf_file(
    "/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_hand_description_right.urdf",
    base_elements=allegro_ring_element,
    active_links_mask=[False] + 4 * [True] + [False],
    symbolic=False,
    name="allegro_ring",
)
allegro_ring_chain.to_json_file(force=True)