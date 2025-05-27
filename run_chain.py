import numpy as np

from ikpy.chain import Chain
from ikpy.utils import plot


# First, let's import the baxter chains
allegro_thumb_chain = Chain.from_json_file("/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_thumb.json")
allegro_index_chain = Chain.from_json_file("/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_index.json")
allegro_middle_chain = Chain.from_json_file("/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_middle.json")
allegro_ring_chain = Chain.from_json_file("/home/hengxuy/PycharmProjects/FastSAM_rob/allegro_hand_description/allegro_ring.json")

# fig, ax = plot.init_3d_figure();
# allegro_thumb_chain.plot([0] * (len(allegro_thumb_chain)), ax)
# allegro_index_chain.plot([0] * (len(allegro_index_chain)), ax)
# allegro_middle_chain.plot([0] * (len(allegro_middle_chain)), ax)
# allegro_ring_chain.plot([0] * (len(allegro_ring_chain)), ax)
# ax.legend()
# plot.show_figure()

# ### Let's try some IK
fig, ax = plot.init_3d_figure();
target = [0.05, 0, 0.05]
target_thumb = [0.07, 0.07, 0]

frame_target = np.eye(4)
frame_target[:3, 3] = target

frame_target_thumb = np.eye(4)
frame_target_thumb[:3, 3] = target_thumb

ik_thumb = allegro_thumb_chain.inverse_kinematics_frame(frame_target_thumb)
ik_index = allegro_index_chain.inverse_kinematics_frame(frame_target)
ik_middle = allegro_middle_chain.inverse_kinematics_frame(frame_target)
ik_ring = allegro_ring_chain.inverse_kinematics_frame(frame_target)

allegro_thumb_chain.plot(ik_thumb, ax, target=target)
allegro_index_chain.plot(ik_index, ax, target=target)
allegro_middle_chain.plot(ik_middle, ax, target=target)
allegro_ring_chain.plot(ik_ring, ax, target=target)
ax.legend()
plot.show_figure()
