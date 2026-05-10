# install: pip install -U langchain-google-genai langchain pandas
import pandas as pd
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv

dotenv.load_dotenv()


def load_and_sample_data(file_path, sample_size=100):
    """Load dataset and return sample"""
    try:
        df = pd.read_csv(file_path)
        sample = df.head(sample_size).to_csv(index=False)
        return sample, df.shape
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def create_cleaning_prompt():
    """Create the data cleaning prompt template"""
    template = """
You are a senior data cleaning expert with 10+ years of experience.
I will give you a dataset sample. Your tasks:

1. **Identify Issues**: Find missing values, inconsistent formatting, invalid data types, outliers
2. **Suggest Fixes**: Recommend specific corrections for each issue
3. **Provide Cleaned Sample**: Show the cleaned version of the sample data
4. **Generate Code**: Create pandas code to clean the full dataset

**Dataset Sample:**
{data}

**Output Format:**
Please provide your response in FOUR clear sections:

A. **PROBLEMS FOUND**
   - Create a table with columns: Column_Name, Issue_Type, Issue_Description, Suggested_Fix
   - List all data quality issues found

B. **CLEANING RECOMMENDATIONS**
   - Detailed explanation of recommended cleaning steps
   - Priority order for fixes

C. **CLEANED SAMPLE DATA**
   - Provide the cleaned sample in CSV format
   - Include header row

D. **PYTHON CLEANING CODE**
   - Generate complete pandas code to clean the full dataset
   - Include functions for each cleaning step
   - Add comments explaining each operation
"""
    return PromptTemplate(input_variables=["data"], template=template)

def setup_gemini_llm():
    """Configure the Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Updated model name
        temperature=0.1,  # Lower temperature for more consistent results
        max_tokens=2000,  # Increased for longer responses
        timeout=60,
    )

def main():
    # Load and sample data
    sample, original_shape = load_and_sample_data("test_data.csv")
    
    if sample is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Original dataset shape: {original_shape}")
    print("Sample data loaded successfully.\n")
    
    # Setup LLM and chain
    prompt = create_cleaning_prompt()
    llm = setup_gemini_llm()
    chain = prompt | llm
    
    try:
        print("Analyzing data with Gemini... This may take a moment.\n")
        result = chain.invoke({"data": sample})
        
        print("=" * 60)
        print("DATA CLEANING ANALYSIS RESULTS")
        print("=" * 60)
        print(result.content)
        
        # Optional: Save results to file
        with open("data_cleaning_analysis.txt", "w") as f:
            f.write(result.content)
        print("\nResults saved to 'data_cleaning_analysis.txt'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

# Enhanced testing for data_viz module
def test_data_viz():
    """Test the data visualization functions"""
    try:
        # Import here to avoid dependency issues if not installed
        from dfx import plot_histograms, plot_correlation_heatmap
        
        # Load your data
        df = pd.read_csv('your_data.csv')
        
        print("Generating visualizations...")
        
        # Histograms with error handling
        try:
            plot_histograms(df, figsize=(12, 8))
            print("✓ Histograms generated successfully")
        except Exception as e:
            print(f"✗ Error generating histograms: {e}")
        
        # Correlation heatmap with error handling
        try:
            plot_correlation_heatmap(df, cmap='viridis')
            print("✓ Correlation heatmap generated successfully")
        except Exception as e:
            print(f"✗ Error generating heatmap: {e}")
            
    except ImportError:
        print("data_viz module not available")
    except FileNotFoundError:
        print("Data file 'your_data.csv' not found")
    except Exception as e:
        print(f"Error in visualization testing: {e}")

if __name__ == "__main__":
    main()
    
    # Uncomment to test visualization module
    # test_data_viz()