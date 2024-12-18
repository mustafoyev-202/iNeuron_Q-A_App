# README.md for iNeuron Q&A Application 🌱

## Overview

Welcome to the **iNeuron Q&A Application**! This web-based question-and-answer system harnesses the power of LangChain
and Streamlit to provide users with an interactive platform for asking questions and receiving accurate answers based on
a curated knowledge base. Utilizing Google Generative AI, this application ensures that responses are both informative
and relevant.

## 🌟 Features

- **Dynamic Knowledge Base Creation**: Easily create a knowledge base from a CSV file containing frequently asked
  questions (FAQs).
- **Natural Language Processing**: Leverages Google Generative AI for nuanced and context-aware responses.
- **User-Friendly Interface**: Built with Streamlit for an engaging and intuitive user experience.

## ⚙️ Requirements

To run this application, ensure you have the following dependencies installed:

- Python 3.7 or higher
- Streamlit
- LangChain
- FAISS
- dotenv
- HuggingFace Transformers

You can install the required packages using pip:

```bash
pip install streamlit langchain langchain_community faiss-cpu python-dotenv transformers
```

## 🛠️ Environment Variables

Before running the application, set up your environment variables in a `.env` file:

```
GOOGLE_API_KEY=your_google_api_key_here
```

## 📁 File Structure

```
/iNeuron-QA-App
│
├── langchain_helper.py  # Logic for creating the vector database and QA chain
├── main.py              # Main entry point for the Streamlit application
└── codebasics_faqs.csv  # CSV file containing example FAQ data
```

## 🚀 Usage Instructions

1. **Clone the Repository**:
   Clone this repository to your local machine.

   ```bash
   git clone https://github.com/mustafoyev-202/iNeuron_Q-A_App.git
   cd iNeuron-QA-App
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add your Google API key.

3. **Run the Application**:
   Start the Streamlit application with the following command:

   ```bash
   streamlit run main.py
   ```

4. **Create Knowledge Base**:
   Click on the "Create Knowledgebase" button to load data from `codebasics_faqs.csv` into the vector database.

5. **Ask Questions**:
   Enter your question in the input field and receive answers based on the knowledge base.

## 💻 Code Explanation

### `langchain_helper.py`

This script is responsible for creating the vector database and setting up the QA chain.

- **create_vector_db()**: Loads data from a CSV file, creates a vector database using FAISS, and saves it locally.
- **get_qa_chain()**: Initializes the QA chain with Google Generative AI and sets up prompt templates for generating
  responses.

### `main.py`

This is the main application script that provides a user interface using Streamlit.

- It initializes the application, allows users to create a knowledge base, and processes user queries to fetch answers
  from the QA chain.

## 🤝 Contributing

We welcome contributions! Feel free to submit issues or pull requests to help improve this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for checking out the iNeuron Q&A Application! We hope you find it useful and engaging. If you have any
questions or feedback, please reach out! 🌟
