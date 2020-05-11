import pybullet as p
import time

c = p.connect(p.GUI)
#p.resetSimulation()

#p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
print(c)
if (c < 0):
  p.connect(p.GUI)

# ハンドモデル[元田/2020/05/09]
HandStartPos = [0, 0, 1]
HandStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

print("assets/hand/grasp_block.xml")
shand = p.loadMJCF("assets/hand/grasp_block.xml")
sh = shand[0]
# p.resetBasePositionAndOrientation(sh, [0.0, 0.0, 0.113124],
#                                  [0.710965, 0.218117, 0.519402, -0.420923])

useRealTimeSimulation = 0

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)

while 1:
  if (useRealTimeSimulation):
    p.setGravity(0, 0, -10)
    sleep(0.01)  # Time in seconds.
  else:
    p.stepSimulation()