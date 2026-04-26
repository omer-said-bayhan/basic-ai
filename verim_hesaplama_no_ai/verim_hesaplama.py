import torch
import torch.nn as nn
import torch.optim as optim
import json
import random

dosya_yolu ="/home/saidbayhan/Masaüstü/ömer_projeler/ai öğrenme projeleri/faz2 sinir ağlari/verim_hesaplama_no_ai/datalar.json"
with open (dosya_yolu,"r") as f:
    data =json.load(f)["veriler"]


class EgitimModeli(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.aikatmanlari = nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),

        )
    def forward(self,x):
        return self.aikatmanlari(x)
model = EgitimModeli(input_dim=4)


hata_olcer = nn.MSELoss()
tamirci = optim.Adam(model.parameters(),lr=0.0001)

for epoch in range (9000):
    ornek = random.choice(data)
    girdi = torch.tensor(ornek["cozulen_sorular"],dtype=torch.float32)
    hedef = torch.tensor([float(ornek["verim"])],dtype=torch.float32)
    tamirci.zero_grad()
    tahmin = model(girdi)
    loss = hata_olcer(tahmin,hedef)
    loss.backward()
    tamirci.step()

def calısmaverim_tahmini(matematik,edebiyat,sosyal,fen):
    girdi= torch.tensor([matematik,edebiyat,sosyal,fen],dtype=torch.float32)
    with torch.no_grad():
        sonuc= model(girdi).item()
    return round(sonuc,1)
print("--------------------verim hesaplama ajanı ---------")
while True:
    try:
            print("\nCihaz Durum Verilerini Girin:")
            m = float(input("Matematik çözülen sayi: "))
            e = float(input("edb çözülen sayi: "))
            s = float(input("sosyal cozulen sayı: "))
            f = float(input("fen cozulen sayı: "))
            
            verim = calısmaverim_tahmini(m, e, s, f)
            
            print("-" * 30)
            print(f"ANALİZ SONUCU: verim {verim}")
            if verim < 20:
                print("⚠️ UYARI: GİT DERS ÇALIŞ")
            print("-" * 30)
    except ValueError:
            break