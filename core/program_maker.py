import random
import json
import pandas as pd

# Function to generate rephrased and shortened variants of questions and responses
def generate_variants(data, num_variants=5, repeat_set=3):
    variants = []
    for _, row in data.iterrows():
        question = row["input"].replace("[QUESTION] ", "").strip()
        response = row["response"].replace("[ANSWER] ", "").strip()

        # Generate variants by lightly rephrasing
        for _ in range(num_variants):
            question_variant = question
            response_variant = response

            # Add simple tweaks to simulate rephrasing (basic transformations for demonstration)
            question_variant = question_variant.replace("PSU", "Pittsburg State University") \
                                               .replace("programs", "courses") \
                                               .replace("offer", "provide")
            
            response_variant = response_variant.replace("PSU", "Pittsburg State University") \
                                               .replace("programs", "academic offerings") \
                                               .replace("fields", "areas of study")
            
            # Shorten the question and response
            question_variant = shorten_question(question_variant)
            response_variant = shorten_response(response_variant)
            
            # Ensure no duplicates and append
            variants.append((question_variant, response_variant))
    
    # Repeat the entire set of variants
    repeated_variants = variants * repeat_set
    return pd.DataFrame(repeated_variants, columns=["Question", "Response"])

# Function to shorten questions
def shorten_question(question):
    replacements = {
        "How many courses does Pittsburg State University provide?": "How many courses at PSU?",
        "What is the total number of courses at Pittsburg State University?": "Total courses at PSU?",
        "Does Pittsburg State University provide online courses?": "Online courses at PSU?",
        "Are online learning options available at Pittsburg State University?": "Online options at PSU?",
        "What types of degrees are available at Pittsburg State University?": "Degrees at PSU?",
        "Does Pittsburg State University provide healthcare courses?": "Healthcare courses at PSU?",
        "What healthcare studies can I pursue at Pittsburg State University?": "Healthcare studies at PSU?",
        "What engineering courses are available at Pittsburg State University?": "Engineering courses at PSU?",
        "Does Pittsburg State University have specialized areas in its courses?": "Specialized areas at PSU?",
        "Can I specialize within Pittsburg State University courses?": "Specializations at PSU?",
        "Are certificate courses available at Pittsburg State University?": "Certificates at PSU?",
        "What business courses are provided at Pittsburg State University?": "Business courses at PSU?",
        "Are there science and technology courses at Pittsburg State University?": "Science and tech courses at PSU?"
    }
    return replacements.get(question, question)

# Function to shorten responses
def shorten_response(response):
    replacements = {
        "Pittsburg State University offers more than 200 academic academic offerings in business, education, engineering, science, humanities, and healthcare.": "PSU offers 200+ courses in various fields.",
        "Pittsburg State University has over 200 academic offerings across various areas of study, including business, healthcare, and engineering.": "PSU has 200+ courses in business, healthcare, and engineering.",
        "Yes, Pittsburg State University provides online degree academic offerings in business, education, and health sciences, allowing for flexible learning schedules.": "PSU offers online degrees in business, education, and health sciences.",
        "Absolutely! Pittsburg State University offers a range of online degrees and courses in areas of study like business, education, and healthcare.": "PSU offers online degrees and courses in business, education, and healthcare.",
        "At Pittsburg State University, you can earn associate, bachelor's, master's, and doctoral degrees, along with certificates and professional development academic offerings.": "PSU offers associate to doctoral degrees and certificates.",
        "Yes, Pittsburg State University provides academic offerings such as Nursing, Healthcare Administration, and Allied Health, some of which are available online.": "PSU offers Nursing, Healthcare Administration, and Allied Health, some online.",
        "Pittsburg State University offers healthcare-related degrees, including Nursing, Healthcare Administration, and Allied Health, with online options for flexibility.": "PSU offers healthcare degrees with online options.",
        "Engineering academic offerings at Pittsburg State University include Mechanical, Civil, and Electrical Engineering, featuring hands-on learning and internships.": "PSU offers Mechanical, Civil, and Electrical Engineering with internships.",
        "Yes, Pittsburg State University offers specialized tracks like Advanced Vehicle Systems, Pre-Medicine Chemistry, and Environmental Chemistry.": "PSU offers tracks in Advanced Vehicle Systems, Pre-Medicine, and Environmental Chemistry.",
        "Certainly! Pittsburg State University provides specialization options in areas such as Pre-Medicine Chemistry, Environmental Chemistry, and Advanced Vehicle Systems.": "PSU offers specializations in Pre-Medicine, Environmental Chemistry, and Advanced Vehicle Systems.",
        "Yes, Pittsburg State University offers certificates in areas of study like Additive Manufacturing, Autism Spectrum Disorders, and Core Studio Media.": "PSU offers certificates in Additive Manufacturing, Autism Spectrum Disorders, and Core Studio Media.",
        "Pittsburg State University provides business academic offerings such as Accounting, an MBA, International Business, and Fraud Examination, with online and on-campus options.": "PSU offers business courses like Accounting, MBA, International Business, and Fraud Examination.",
        "Yes, Pittsburg State University features academic offerings in Biology, Chemistry, and Computer Science, as well as Electronics Engineering.": "PSU offers courses in Biology, Chemistry, Computer Science, and Electronics Engineering.",
        "For detailed information, visit Pittsburg State University's official website to see program descriptions, degree pathways, and admission requirements.": "Visit PSU's website for program details.",
        "Certainly! Pittsburg State University offers courses in areas of study like Biology, Computer Science, and Electronics Engineering, with specialized tracks available.": "PSU offers courses in Biology, Computer Science, and Electronics Engineering.",
        "To explore Pittsburg State University's academic offerings, visit the official website for up-to-date program details and application guidelines.": "Visit PSU's website for course details."
    }
    return replacements.get(response, response)

# Load the programs.json file to training_data
with open('./programs.json') as f:
    data = json.load(f)
training_data = pd.DataFrame(data)

# Generate variants for the training data
augmented_training_data = generate_variants(training_data)

# Display a sample of the augmented dataset
print(augmented_training_data.head(), augmented_training_data.shape)

# Save the augmented dataset to a TSV file without headers
output_file_path = './augmented_training_data.tsv'
augmented_training_data.to_csv(output_file_path, sep='\t', index=False, header=False)

# print(output_file_path)
print("Data augmentation complete!")