# -*- coding: utf-8 -*-
"""MNIST: FP32, mixed FP16/FP32 и FP16 inference. Запуск из Google Colab."""
import torch, torchvision, numpy as np, pandas as pd, matplotlib.pyplot as plt, time, copy, random, gc
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('Включите GPU в Google Colab: Runtime → Change runtime type → GPU')
DEVICE = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# ============================================================

BATCH_SIZE=128
EPOCHS=2  # увеличьте до 3–5 для более устойчивой оценки точности
CHANNELS=500
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
train_ds=datasets.MNIST('./data',train=True,download=True,transform=transform)
test_ds=datasets.MNIST('./data',train=False,download=True,transform=transform)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
test_loader=DataLoader(test_ds,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,CHANNELS,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(CHANNELS,CHANNELS,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(CHANNELS,CHANNELS,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(CHANNELS*4,500),nn.ReLU(),nn.Linear(500,10))
    def forward(self,x):return self.classifier(self.features(x))
base=BigCNN(); initial=copy.deepcopy(base.state_dict()); del base
print('Параметров:',sum(v.numel() for v in initial.values()))


# ============================================================

# Визуализация 20 примеров из обучающей выборки MNIST
fig, axes = plt.subplots(4, 5, figsize=(11, 9))
for idx, ax in enumerate(axes.flat):
    image, label = train_ds[idx]
    image = image.squeeze(0) * 0.3081 + 0.1307  # отменяем нормализацию
    ax.imshow(image.clamp(0, 1).numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Цифра: {label}')
    ax.axis('off')
plt.suptitle('Примеры рукописных цифр MNIST', fontsize=15)
plt.tight_layout()
plt.show()


# ============================================================

def evaluate(model, mixed=False):
    model.eval(); correct=total=0
    with torch.inference_mode():
        for x,y in test_loader:
            x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=mixed):
                logits=model(x)
            correct+=(logits.argmax(1)==y).sum().item(); total+=y.numel()
    return 100*correct/total

def train_mode(mode):
    gc.collect(); torch.cuda.empty_cache()
    model=BigCNN().to(DEVICE); model.load_state_dict(initial)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    mixed=(mode=='Mixed FP16/FP32')
    scaler=torch.amp.GradScaler('cuda',enabled=mixed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start=time.perf_counter()
    losses=[]
    for epoch in range(EPOCHS):
        model.train(); running=0
        for x,y in train_loader:
            x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=mixed):
                logits=model(x); loss=nn.functional.cross_entropy(logits,y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running+=loss.item()
        losses.append(running/len(train_loader))
        print(f'{mode}, эпоха {epoch+1}/{EPOCHS}, loss={losses[-1]:.4f}')
    torch.cuda.synchronize(); train_seconds=time.perf_counter()-start
    accuracy=evaluate(model,mixed)
    peak_mb=torch.cuda.max_memory_allocated()/1024**2
    weights_mb=sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    result={'Режим':mode,'Время обучения, с':train_seconds,'Точность, %':accuracy,'Пик GPU памяти, МиБ':peak_mb,'Веса, МиБ':weights_mb}
    del optimizer,model,scaler; gc.collect(); torch.cuda.empty_cache()
    return result,losses
results=[]; history={}
for mode in ('FP32','Mixed FP16/FP32'):
    r,h=train_mode(mode); results.append(r); history[mode]=h


# ============================================================

# Для сопоставимого прямого прохода обучаем FP32-модель отдельно из тех же начальных весов.
# При желании можно заменить на сохранённую модель из первого эксперимента.
def train_reference():
    model=BigCNN().to(DEVICE); model.load_state_dict(initial)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for x,y in train_loader:
            x=x.to(DEVICE);y=y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x),y)
            loss.backward();opt.step()
    return model
reference=train_reference()
reference.eval(); half_model=copy.deepcopy(reference).half().eval()
del reference; gc.collect();torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
correct=total=0
with torch.inference_mode():
    for x,y in test_loader:
        x=x.to(DEVICE,dtype=torch.float16);y=y.to(DEVICE)
        pred=half_model(x).argmax(1)
        correct+=(pred==y).sum().item();total+=y.numel()
torch.cuda.synchronize(); inference_s=time.perf_counter()-start
results.append({'Режим':'FP16 (только inference)','Время обучения, с':float('nan'),'Точность, %':100*correct/total,'Пик GPU памяти, МиБ':torch.cuda.max_memory_allocated()/1024**2,'Веса, МиБ':sum(p.numel()*p.element_size() for p in half_model.parameters())/1024**2})
print('Время FP16 inference, с:',round(inference_s,3))
del half_model;gc.collect();torch.cuda.empty_cache()


# ============================================================

df=pd.DataFrame(results)
display(df.round(3))
speedup=df.loc[0,'Время обучения, с']/df.loc[1,'Время обучения, с']
accuracy_diff=df.loc[1,'Точность, %']-df.loc[0,'Точность, %']
memory_ratio=df.loc[0,'Пик GPU памяти, МиБ']/df.loc[1,'Пик GPU памяти, МиБ']
print(f'Ускорение mixed precision относительно FP32: {speedup:.2f}×')
print(f'Разница точности (mixed − FP32): {accuracy_diff:+.2f} процентных пункта')
print(f'Отношение пиков выделенной памяти FP32/mixed: {memory_ratio:.2f}×')
fig,axes=plt.subplots(1,3,figsize=(16,4))
train_df=df.iloc[:2]
for ax,col in zip(axes,['Время обучения, с','Точность, %','Пик GPU памяти, МиБ']):
    ax.bar(train_df['Режим'],train_df[col]);ax.set_title(col);ax.tick_params(axis='x',rotation=12)
plt.tight_layout();plt.show()
plt.figure(figsize=(7,4))
for mode,losses in history.items():plt.plot(range(1,len(losses)+1),losses,marker='o',label=mode)
plt.xlabel('Эпоха');plt.ylabel('Средняя ошибка');plt.legend();plt.grid(True);plt.show()
print('ВЫВОД:')
print(f'На GPU {torch.cuda.get_device_name(0)} смешанная точность показала ускорение {speedup:.2f}× по сравнению с FP32.')
print(f'Изменение точности составило {accuracy_diff:+.2f} п.п. Память весов FP16 вдвое меньше FP32.')
print('Полный FP16 проверен только на прямом проходе; его показатели памяти нельзя напрямую сравнивать с обучением.')
