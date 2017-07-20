import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

#Plotting loss
with open(log_dir+"\\training_log_corrected.txt") as f:
    log_data = f.readlines()
    

train_steps = []
disc_real_loss = []
disc_fake_loss = []
GAN_loss = []
for log in log_data:
    train_steps.append(int(log.split(' ')[1]))
    disc_real_loss.append(float(log.split(' ')[5]))
    disc_fake_loss.append(float(log.split(' ')[8]))
    GAN_loss.append(float(log.split(' ')[11][0:-1]))


#takes too long in my comp; didn't use
#x_smooth = np.linspace(np.array(train_steps).min(), np.array(train_steps).max(), len(train_steps)/10)
#y_smooth = spline(train_steps, disc_real_loss, x_smooth)
#plt.plot(x_smooth,y_smooth,label="Discriminator real loss")

"""
#all 3 in one plot
plt.plot(train_steps,disc_real_loss,label="Discriminator real loss")
plt.plot(train_steps,disc_fake_loss,label="Discriminator fake loss")
plt.plot(train_steps,GAN_loss,label="Generator loss")

plt.xlabel('Train steps')
plt.ylabel('Loss')
plt.show()
"""

#real vs fake loss plot
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(train_steps, disc_real_loss, '-', label="Discriminator real loss")
lns2 = ax.plot(train_steps, disc_fake_loss, '-', label="Discriminator fake loss")

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Train steps")
ax.set_ylabel("Loss")
plt.show()

#gen loss plot
fig = plt.figure()
ax = fig.add_subplot(111)

lns3 = ax.plot(train_steps, GAN_loss, '-', label="Generator loss")
lns = lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Train steps")
ax.set_ylabel("Loss")
plt.show()