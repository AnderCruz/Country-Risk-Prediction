from data.sources import WorldBankSource

def main():

    print("=" * 70)
    print("UPDATING DATA")
    print("=" * 70)

    source = WorldBankSource()
    source.download()

    print("\nData update completed.")


if __name__ == "__main__":
    main()