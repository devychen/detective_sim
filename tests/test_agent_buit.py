# main.py
from agent._archive.holmes_agent import create_holmes_agent

def main():
    holmes = create_holmes_agent()
    input_text = "You arrive at the scene of a murder. What do you see and deduce first?"
    response = holmes.run(input_text)
    print("\n🔍 Sherlock Holmes says:\n")
    print(response)

if __name__ == "__main__":
    main()
