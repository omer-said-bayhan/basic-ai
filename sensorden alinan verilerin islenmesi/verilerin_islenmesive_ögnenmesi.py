import torch
import torch.nn as nn
import torch.optim as optim
import json 
import random





dosya_yolu ="/home/saidbayhan/Masaüstü/ömer projeler/ai öğrenme projeleri/faz2 sinir ağlari/sensorden alinan verilerin islenmesi/sensor_data.json"
with open(dosya_yolu,"r") as f:
    data = json.load(f)["veriler"]

class OgrenmeModeli(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.katmanlar = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.katmanlar(x)
    
model =OgrenmeModeli(input_dim=4)



hata_olcer = nn.MSELoss()# bizim verdiğimiz cevap ile json dosyasındaki cevap arasında ne kadar fark olduğunu hesaplayıp ölçüp yapılıyor
tamirci = optim.Adam(model.parameters(),lr=0.001)# optim.Adam yapy zeka öğretmwni olarak görev alıyor lr yi eğer yuksek veririsen bazı seylerin üstünden atlar ve az öğrenir cok veririsende cok bekleriz

for epoch in range (2000):
    ornek = random.choice(data)

    girdi = torch.tensor(ornek["sensorler"],dtype=torch.float32)
    hedef = torch.tensor([float(ornek["kalan_omur"])],dtype=torch.float32)


    tamirci.zero_grad()# eksi hatalaerdan kalan verileri siler
    tahmin = model(girdi)# bilgi seviyesiyle bir tahminde bulunur
    loss = hata_olcer(tahmin,hedef)# mesloss ile bir hatayı sayısal değere dönusturur 
    loss.backward()# ilk katmana geri gidip kimin hata yaptıgını bulur  
    tamirci.step()# hata yapanı değiştirir

def omur_tahmin_yap(sicaklik,titresim,basinc,rpm):
    girdi= torch.tensor([sicaklik,titresim,basinc,rpm],dtype=torch.float32)
    with torch.no_grad():
        sonuc = model(girdi).item()# cevap float olur
    return round(sonuc,1)
print("-----Mühendislij bakım asistanı hazır--")
while True:
    try:
            print("\nCihaz Durum Verilerini Girin:")
            s = float(input("Sıcaklık (°C): "))
            t = float(input("Titreşim (mm/s): "))
            b = float(input("Basınç (bar): "))
            r = float(input("Devir (RPM): "))
            
            gun = omur_tahmin_yap(s, t, b, r)
            
            print("-" * 30)
            print(f"ANALİZ SONUCU: Bu cihazın tahmini {gun} günlük ömrü kaldı.")
            if gun < 20:
                print("⚠️ UYARI: ACİL BAKIM GEREKLİ!")
            print("-" * 30)
    except ValueError:
            break