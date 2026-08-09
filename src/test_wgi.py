from data.sources.wgi import WGISource


source = WGISource()

governance = source.process()

print("\nGovernance dataset")
print("=" * 60)

print(governance.head())

print("\nColumns:")
print(governance.columns.tolist())

print("\nShape:")
print(governance.shape)

print("\nMissing values:")
print(governance.isna().sum())