from speckle import speckle

def example_1() -> None:
    """
    Generate a simple 500x500 speckle pattern with a tiny amount of randomness
    """
    print("Generating a 500 by 500 speckle pattern to img.png")
    # speckle.speckle(w="500", l="500")
    speckle.speckle()
    print("Done!")

if __name__ == "__main__":
    example_1()
