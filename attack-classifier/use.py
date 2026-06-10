# import os
# import pandas as pd
# import json
# import tensorflow as tf
# import tensorflow_hub as hub

# # Define the USE class as in your original example
# class USE(object):
#     def __init__(self, cache_path):
#         super(USE, self).__init__()
#         os.environ['TFHUB_CACHE_DIR'] = cache_path
#         module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
#         self.embed = hub.Module(module_url)
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.build_graph()
#         self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

#     def build_graph(self):
#         self.sts_input1 = tf.placeholder(tf.string, shape=(None))
#         self.sts_input2 = tf.placeholder(tf.string, shape=(None))

#         sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
#         sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
#         self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
#         clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
#         self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

#     def semantic_sim(self, sents1, sents2):
#         scores = self.sess.run(
#             [self.sim_scores],
#             feed_dict={
#                 self.sts_input1: sents1,
#                 self.sts_input2: sents2,
#             })
#         return scores[0].tolist()  # Return similarity as a list of floats

# # Function to process JSON files and compute similarity
# def process_json_files(directory_path, cache_path):
#     # Initialize USE model
#     use_model = USE(cache_path)
    
#     # Iterate over all JSON files in the directory
#     for json_file in os.listdir(directory_path):
#         if json_file.endswith('.json'):
#             # Construct full file path
#             file_path = os.path.join(directory_path, json_file)
            
#             # Read the JSON file as a pandas DataFrame
#             df = pd.read_json(file_path)
            
#             # Extract sourceDocs and advDocs
#             df['sourceDocs'] = df['sourceDocs'].apply(lambda x: ' '.join(x))
#             df['advDocs'] = df['advDocs'].apply(lambda x: ' '.join(x))
#             source_docs = df['sourceDocs'].tolist()
#             adv_docs = df['advDocs'].tolist()

#             # Ensure the lengths of sourceDocs and advDocs match
#             if len(source_docs) != len(adv_docs):
#                 print(f"Warning: sourceDocs and advDocs have different lengths in {json_file}")
#                 continue

#             # Compute similarity using USE
#             similarity_scores = use_model.semantic_sim(source_docs, adv_docs)
#             similarity_scores = round(sum(similarity_scores)/len(similarity_scores),6)
            
#             # Print the result
#             print(f"File: {json_file}")
#             print(f"Similarity scores: {similarity_scores}")

# # Usage example:
# # directory_path = './json_files'  # Replace with your JSON directory
# # cache_path = './'  # Specify your cache path for TF Hub

# directory_path = input('Type with your JSON directory: ')
# cache_path = input('Specify your cache path for TF Hub: ')

# # Call the function to process all JSON files and compute similarity
# process_json_files(directory_path, cache_path)


import os
import pandas as pd
import json
import tensorflow as tf
import tensorflow_hub as hub

# Define the USE class as in your original example
class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores[0].tolist()  # Return similarity as a list of floats

# Function to process JSON files and compute similarity
def process_json_files(directory_path, cache_path):
    # Initialize USE model
    use_model = USE(cache_path)

    # Open the output file in write mode
    output_file_path = os.path.join(directory_path, 'use_compute.txt')
    with open(output_file_path, 'w') as output_file:
        # Iterate over all JSON files in the directory
        for json_file in os.listdir(directory_path):
            if json_file.endswith('.json'):
                # Construct full file path
                file_path = os.path.join(directory_path, json_file)

                # Read the JSON file as a pandas DataFrame
                df = pd.read_json(file_path)

                # Extract sourceDocs and advDocs
                df['sourceDocs'] = df['sourceDocs'].apply(lambda x: ' '.join(x))
                df['advDocs'] = df['advDocs'].apply(lambda x: ' '.join(x))
                source_docs = df['sourceDocs'].tolist()
                adv_docs = df['advDocs'].tolist()

                # Ensure the lengths of sourceDocs and advDocs match
                if len(source_docs) != len(adv_docs):
                    output_file.write(f"Warning: sourceDocs and advDocs have different lengths in {json_file}\n")
                    continue

                # Compute similarity using USE
                similarity_scores = use_model.semantic_sim(source_docs, adv_docs)
                similarity_mean = round(sum(similarity_scores) / len(similarity_scores), 6)

                # Write the result to the file
                output_file.write(f"File: {json_file}\n")
                output_file.write(f"Similarity scores: {similarity_mean}\n\n")
                print(f"File: {json_file}")
                print(f"Similarity scores: {similarity_mean}")
                print(f"File: {json_file} processed. Similarity score written to use_compute.txt")

# Usage example:
# directory_path = './json_files'  # Replace with your JSON directory
# cache_path = './'  # Specify your cache path for TF Hub

directory_path = input('Type your JSON directory: ')
cache_path = input('Specify your cache path for TF Hub: ')

# Call the function to process all JSON files and compute similarity
process_json_files(directory_path, cache_path)
