import numpy as np, pandas as pd, json
rng = np.random.default_rng(42)
n = 1000
cities = ["Mumbai","Delhi","Chennai","Kolkata","Bengaluru","Hyderabad","Pune","Ahmedabad"]
modes = ["Air","Road","Rail","Sea"]
df = pd.DataFrame({
 "Shipment_ID": np.arange(1001,1001+n),
 "Order_Date": pd.to_datetime("2025-01-01")+pd.to_timedelta(rng.integers(0,300,n),unit="D"),
 "Origin": rng.choice(cities,n),
 "Destination": rng.choice(cities,n),
 "Shipping_Mode": rng.choice(modes,n,p=[.15,.5,.25,.10]),
 "Distance_km": rng.normal(1100,450,n).clip(120,2600).round(1),
 "Weight_kg": rng.gamma(4,60,n).round(1),
 "Inventory_Level": rng.integers(20,500,n).astype(float),
})
speed={"Air":800,"Road":45,"Rail":60,"Sea":30}
df["Transit_Days"]=(df.Distance_km/df.Shipping_Mode.map(speed)/ (1 if False else 1)*1).clip(lower=0)
df["Transit_Days"]=(np.ceil(df.Distance_km/df.Shipping_Mode.map({"Air":900,"Road":450,"Rail":550,"Sea":300}))+rng.integers(0,3,n)).astype(float)
df["Ship_Date"]=df.Order_Date+pd.to_timedelta(rng.integers(0,3,n),unit="D")
df["Delivery_Date"]=df.Ship_Date+pd.to_timedelta(df.Transit_Days,unit="D")
rate=df.Shipping_Mode.map({"Air":9,"Road":3,"Rail":2,"Sea":1.5})
df["Freight_Cost"]=(df.Distance_km*rate*(0.6+df.Weight_kg/400)+rng.normal(0,300,n)).clip(lower=200).round(2)
df["Delivery_Status"]=np.where(rng.random(n)<.82,"On Time","Delayed")
# --- inject quality issues (simulated raw data) ---
raw=df.copy()
for c,f in [("Weight_kg",.06),("Freight_Cost",.05),("Inventory_Level",.08),("Transit_Days",.04)]:
    raw.loc[rng.choice(n,int(n*f),replace=False),c]=np.nan
raw.loc[rng.choice(n,12,replace=False),"Freight_Cost"]*=15   # outliers
raw.loc[rng.choice(n,8,replace=False),"Weight_kg"]*=12
raw.loc[rng.choice(n,10,replace=False),"Distance_km"]=-1*raw.Distance_km   # invalid negatives
mi=rng.choice(n,90,replace=False)
raw.loc[mi,"Shipping_Mode"]=raw.loc[mi,"Shipping_Mode"].str.lower()
mj=rng.choice(n,40,replace=False)
raw.loc[mj,"Origin"]=raw.loc[mj,"Origin"].str.upper()+" "
raw=pd.concat([raw,raw.sample(25,random_state=1)],ignore_index=True)   # duplicates
raw.to_csv("logistics_raw.csv",index=False)

R={}
R["raw_shape"]=list(raw.shape)
R["missing"]=raw.isna().sum()[lambda s:s>0].to_dict()
R["dups"]=int(raw.duplicated().sum())
R["neg_dist"]=int((raw.Distance_km<0).sum())
R["mode_vals"]=sorted(raw.Shipping_Mode.unique().tolist())
R["origin_n"]=int(raw.Origin.nunique())
# --- cleaning ---
d=raw.drop_duplicates().copy()
d["Shipping_Mode"]=d.Shipping_Mode.str.strip().str.title()
for c in ["Origin","Destination"]: d[c]=d[c].str.strip().str.title()
d.loc[d.Distance_km<0,"Distance_km"]=d.Distance_km.abs()
R["after_dedup"]=len(d); R["mode_after"]=sorted(d.Shipping_Mode.unique().tolist()); R["origin_after"]=int(d.Origin.nunique())
# outliers IQR
def iqr(s):
    q1,q3=s.quantile([.25,.75]); i=q3-q1; return q1-1.5*i,q3+1.5*i
R["outliers"]={}
for c in ["Freight_Cost","Weight_kg"]:
    lo,hi=iqr(d[c].dropna()); m=(d[c]<lo)|(d[c]>hi)
    R["outliers"][c]={"lo":round(lo,1),"hi":round(hi,1),"n":int(m.sum())}
    d[c+"_flag"]=m
    d[c]=d[c].clip(lo,hi)  # winsorise (NaN preserved)
# impute
for c in ["Weight_kg","Freight_Cost","Transit_Days"]:
    d[c]=d[c].fillna(d.groupby("Shipping_Mode")[c].transform("median"))
d["Inventory_Level"]=d.Inventory_Level.fillna(d.Inventory_Level.median())
R["missing_after"]=int(d[["Weight_kg","Freight_Cost","Inventory_Level","Transit_Days"]].isna().sum().sum())
# normalise
num=["Distance_km","Weight_kg","Freight_Cost","Inventory_Level","Transit_Days"]
d_mm=(d[num]-d[num].min())/(d[num].max()-d[num].min())
d_z=(d[num]-d[num].mean())/d[num].std()
R["mm_range"]=[float(d_mm.min().min()),float(d_mm.max().max())]
R["z_mean"]=round(float(d_z.mean().abs().max()),6); R["z_std"]=round(float(d_z.std().mean()),3)
d=pd.concat([d,d_mm.add_suffix("_minmax")],axis=1)
d=pd.get_dummies(d,columns=["Shipping_Mode"],prefix="Mode")
d.to_csv("logistics_clean.csv",index=False)
R["final_shape"]=list(d.shape)
R["describe_raw"]=raw[["Distance_km","Weight_kg","Freight_Cost"]].describe().round(1).loc[["mean","std","min","max"]].to_dict()
R["describe_clean"]=d[["Distance_km","Weight_kg","Freight_Cost"]].describe().round(1).loc[["mean","std","min","max"]].to_dict()
json.dump(R,open("results.json","w"),indent=1,default=str)
print(json.dumps(R,indent=1,default=str))
