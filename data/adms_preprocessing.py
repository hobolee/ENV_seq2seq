import torch
import matplotlib.pyplot as plt

adms = torch.load(r'C:\Users\lihaobo\Downloads\data\data_no2\data_adms.pt')
adms = adms.view(-1, 240, 305)
plt.figure()
plt.contourf(adms[20+24+72, ...])
plt.colorbar()
plt.show()

a = torch.max(adms)
b = torch.min(adms)
pass