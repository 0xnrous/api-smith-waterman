from flask import Blueprint, render_template, request, jsonify,Flask
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import requests
import sys
#from script import SequencesAnalyzer

app = Flask(__name__)

# Import ScoringSystems and SequencesAnalyzer here
API_URL = 'https://dna-testing-system.onrender.com/EisaAPI'
SCORES_CSV = './scores.csv'
EDIT_COST_CSV = './edit_cost.csv'

class ScoringSystems:
    '''Responsible for returning proper scoring/edit cost values for any letter combination'''

    def __init__(self, match: int=1, mismatch: int=-1, gap: int=-1) -> None:
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.custom_scoring = None

    def load_csv(self, filename: str) -> None:
        self.custom_scoring = pd.read_csv(filename, header=0, index_col=0, sep=' ')

    def _default_scoring(self, a: str, b: str) -> int:
        '''Checks if there's a match/mismatch/gap for letters a and b'''
        if a == b:
            return self.match
        elif a == '-' or b == '-':
            return self.gap
        return self.mismatch

    def score(self, a: str, b: str) -> int:
        '''This method should be used by algorithms'''
        assert isinstance(a, str) and isinstance(b, str)
        assert len(a) == 1 and len(b) == 1

        # Use CSV file
        if self.custom_scoring is not None:
            try:
                return self.custom_scoring[a][b]
            except KeyError:
                print(f'WARNING: Key not found. You using defaults: {self.__str__}')
                # In case some letter was not present in CSV file, use default scoring value
                return self._default_scoring(a, b)
        # Simply use match/mismatch/gap values
        else:
            return self._default_scoring(a, b)
        
    def __str__(self):
        if self.custom_scoring is not None:
            # Use pandas DataFrame __str__ representation
            return str(self.custom_scoring)
        return f'Match: {self.match}, Mismatch: {self.mismatch}, Gap: {self.gap}'
    
 

class SequencesAnalyzer:

    # Useful for visualization
    traceback_symbols = {
        0: '↖',
        1: '↑',
        2: '←',
        3: '•'
    }

    def __init__(self, seq_a: str, seq_b: str, load_csv: bool = False) -> None:
        self.seq_a = seq_a
        self.seq_b = seq_b

        self.scoring_sys = ScoringSystems(match=2, mismatch=-1, gap=-2)
        self.edit_cost_sys = ScoringSystems(match=0, mismatch=1, gap=1)

        if load_csv:
            self.scoring_sys.load_csv('scores.csv')
            self.edit_cost_sys.load_csv('edit_cost.csv')


    def local_alignment(self) -> Tuple[str, str]:
        result: Dict[str, Any] = self.smith_waterman_algorithm()
        alignment_a, alignment_b = self._traceback(
            result_matrix=result['result_matrix'],
            traceback_matrix=result['traceback_matrix'],
            start_pos=result['score_pos'],
            global_alignment=False)
        # Calculate the maximum possible score
        max_possible_score = min(len(self.seq_a), len(self.seq_b)) * self.scoring_sys.match

        # Get the actual alignment score
        alignment_score = result['score']

        # Calculate the percentage score
        score_percentage = (alignment_score / max_possible_score) * 100

        # Update the result dictionary with the percentage score
        result['score'] = score_percentage

        print(
            f"[Local Alignment Percentage] Score={result['score']}\n"
            #f"Result:\n {result['result_matrix']}\n"
            #f"Traceback:\n {result['traceback_matrix']}\n"
            #f"Alignment:\n {alignment_a}\n {alignment_b}\n"
        )
        return alignment_a, alignment_b , score_percentage

    def similarity(self) -> int:
        result = self.needleman_wunsch_algorithm(minimize=False)
        
        print(
            f"[Similarity] Score={result['score']}\n"
            #f"{result['result_matrix']}\n"
            #f"{result['traceback_matrix']}\n"
        )
        return result['score']
    

    

    def edit_distance(self) -> int:
        result = self.needleman_wunsch_algorithm(minimize=True)

        print(
            #f"[Edit distance] Cost={result['score']}\n"
            #f"{result['result_matrix']}\n"
            #f"{result['traceback_matrix']}\n"
        )
        return result['score']

    def needleman_wunsch_algorithm(self, minimize: bool = False, alignment_cal: bool = False) -> Dict[str, Any]:
        '''
        `minimize` - set to True when calculating edit distance
        `alignment_cal` - set to True when calculating global alignment
        '''
        # 1. Prepare dimensions (required additional 1 column and 1 row)
        rows, cols = len(self.seq_a) + 1, len(self.seq_b) + 1

        # 2. Initialize matrices
        # Use grid/matrix as graph-like acyclic digraph (array cells are vertices)
        H = np.zeros(shape=(rows, cols), dtype=int)
        traceback = np.zeros(shape=(rows, cols), dtype=np.dtype('U5'))

        # 3. msadkjnsadkjn;sdf
        gapH = np.zeros(shape=(rows, cols), dtype=int)

        if minimize:
            # Edit cost calculation
            score_func = self.edit_cost_sys.score
        else:
            # Similarity calculation
            score_func = self.scoring_sys.score

        if alignment_cal:
            # Global alignment calculation -> 1st row and column need to have negative values
            sign = self.scoring_sys.gap
        else:
            # Similarity or edit cost calculation -> 1st first row and column values need to be positive
            sign = 1

        # Put sequences' letters into 1st row and 1st column (for better visualization)
        traceback[0, 1:] = np.array(list(self.seq_b), dtype=str)
        traceback[1:, 0] = np.array(list(self.seq_a), dtype=str)
    
        # 3. Top row and leftmost column, like: 0, 1, 2, 3, etc.
        H[0, :] = np.arange(start=0, stop=sign*cols, step=sign)
        H[:, 0] = np.arange(start=0, stop=sign*rows, step=sign)

        for row in range(1, rows):
            for col in range(1, cols):
                # Current pair of letters from sequence A and B
                a = self.seq_a[row - 1]
                b = self.seq_b[col - 1]

                leave_or_replace_letter = H[row -
                    1, col - 1] + score_func(a, b)

                if gapH[row - 1, col] == 0:
                    score = score_func('-', b)
                else:
                    score = 0
                delete_indel = H[row - 1, col] + score

                if gapH[row, col - 1] == 0:
                    score = score_func(a, '-')
                else:
                    score = 0
                insert_indel = H[row, col - 1] + score

                scores = [leave_or_replace_letter, delete_indel, insert_indel]

                if minimize:
                    best_action = np.argmin(scores)
                else:
                    best_action = np.argmax(scores)
                    if best_action in [1, 2]:
                        gapH[row, col] = True

                H[row, col] = scores[best_action]
                traceback[row, col] = self.traceback_symbols[best_action]
        
        #print(gapH)
        return {
            'result_matrix': H,
            'traceback_matrix': traceback,
            'score': H[-1, -1],                 # Always right-bottom corner
            'score_pos': (rows - 1, cols - 1)   # as above...
        }


    def smith_waterman_algorithm(self) -> Dict[str, Any]:
        '''
        Note: Smith-Waterman and Needleman-Wunsch algorithms
        are very similar, but because there are small differences,
        they are meant to be separated.
        '''
        # 1. Prepare dimensions (required additional 1 column and 1 row)
        rows, cols = len(self.seq_a) + 1, len(self.seq_b) + 1

        # 2. Initialize matrices
        # Use grid/matrix as graph-like acyclic digraph (array cells are vertices)
        H = np.zeros(shape=(rows, cols), dtype=int)
        traceback = np.zeros(shape=(rows, cols), dtype=np.dtype('U5'))

        # Difference 1: 1st row and 1st column are already zeroed

        # Put sequences' letters into first row and first column (better visualization)
        traceback[0, 1:] = np.array(list(self.seq_b), dtype=str)
        traceback[1:, 0] = np.array(list(self.seq_a), dtype=str)

        # 3. Top row and leftmost colum are already 0
        for row in range(1, rows):
            for col in range(1, cols):
                # Alias: current pair of letters
                a = self.seq_a[row - 1]
                b = self.seq_b[col - 1]

                score_func = self.scoring_sys.score
                leave_or_replace_letter = H[row -
                    1, col - 1] + score_func(a, b)
                delete_indel = H[row - 1, col] + score_func('-', b)
                insert_indel = H[row, col - 1] + score_func(a, '-')

                # Difference 2: That additional 0 is required (ignore negative values)
                scores = [leave_or_replace_letter,
                    delete_indel, insert_indel, 0]
                best_action = np.argmax(scores)

                H[row, col] = scores[best_action]
                traceback[row, col] = self.traceback_symbols[best_action]

        return {
            'result_matrix': H,
            'traceback_matrix': traceback,
            'score': H.max(),
            # Force numpy to return last result
            # Source: (Step 2: Backtracing) https://tiefenauer.github.io/blog/smith-waterman
            'score_pos': np.unravel_index(np.argmax(H, axis=None), H.shape)
        }

    def _traceback(self, result_matrix, traceback_matrix, start_pos: Tuple[int, int], global_alignment: bool) -> Tuple[str, str]:
        seq_a_aligned = ''
        seq_b_aligned = ''

        # 1. Select starting point
        row, col = start_pos

        if global_alignment:
            # Terminate when top left corner (0,0) is reached (end of path)
            end_condition_reached = lambda row, col: row == 0 and col == 0
        else:
            # Terminate when 0 is reached
            end_condition_reached = lambda row, col: result_matrix[row, col] == 0

        while not end_condition_reached(row, col):
            symbol = traceback_matrix[row, col]
            if row == 0:
                symbol = '←'
            if col == 0:
                symbol = '↑'
            # Use arrows to navigate and collect letters (in reversed order)
            # Shift/reverse indexes by one beforehand (we want to get the letter that arrow points to)
            if symbol == '↖':
                row -= 1
                col -= 1
                letter_a, letter_b = self.seq_a[row], self.seq_b[col]
            elif symbol == '↑':
                row -= 1
                letter_a, letter_b = self.seq_a[row], '-'
            elif symbol == '←':
                col -= 1
                letter_a, letter_b = '-', self.seq_b[col]

            # Acumulate letter (in reversed order)
            seq_a_aligned += letter_a
            seq_b_aligned += letter_b

        # Reverse strings (traceback goes from bottom-right to top-left)
        return seq_a_aligned[::-1], seq_b_aligned[::-1]

# def main():
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <sequence_a_file> <sequence_b_file> [--summary as --S] [--similarity as --s] [--edit-distance as --e ] [--alignment <global/local>]")
#         return

#     sequence_a_file = sys.argv[1]
#     sequence_b_file = sys.argv[2]
#     #summary = '-S' in sys.argv
#     summary = False
#     similarity = True
#     #similarity = '-s' in sys.argv
#     #edit_distance = '-e' in sys.argv
#     # alignment = None
#     # if '--alignment' in sys.argv:
#     #     alignment_index = sys.argv.index('--alignment')
#     #     alignment = sys.argv[alignment_index + 1] if alignment_index + 1 < len(sys.argv) else None

#     # Load CSV files
#     scoring_sys = ScoringSystems()
#     scoring_sys.load_csv(SCORES_CSV)
#     edit_cost_sys = ScoringSystems()
#     edit_cost_sys.load_csv(EDIT_COST_CSV)

#     with open(sequence_a_file, 'r') as file_a, open(sequence_b_file, 'r') as file_b:
#         sequence_a = ''.join(line.strip()[:500] for line in file_a if not line.startswith('>'))
#         sequence_b = ''.join(line.strip()[:500] for line in file_b if not line.startswith('>'))

#     analyzer = SequencesAnalyzer(sequence_a, sequence_b )

#     if summary:
#         pass 
#     elif similarity:
#         #analyzer.similarity()
#         alignment_a, alignment_b = analyzer.local_alignment()
#         if analyzer.similarity() == 1000:
#             print("DNA matched")
#         else:
#             print("Not matched")

def process_uploaded_file(file):
    data = b''  # Initialize as bytes
    total_length = 0
    for line in file:
        line = line.strip()
        if line.startswith(b'>'):  # Compare with byte string
            continue  # Skip header lines
        remaining_length = 500 - total_length
        if remaining_length <= 0:
            break  # Exit loop if the limit is reached
        # Concatenate the line directly
        data += line[:remaining_length]
        total_length += len(data)
    return data.decode()  # Decode the result back to string


def retrieve_dna_sequence_from_file(file):
    data = b''  # Initialize as bytes
    total_length = 0
    for line in file:
        line = line.strip()
        if line.startswith(b'>'):  # Compare with byte string
            continue  # Skip header lines
        remaining_length = 1000 - total_length
        if remaining_length <= 0:
            break  # Exit loop if the limit is reached
        # Concatenate the line directly
        data += line[:remaining_length]
        total_length += len(data)
    return data.decode()  # Decode the result back to string

def retrieve_api_data(API_URL):
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            api_data = response.json()
            if not isinstance(api_data, dict) or 'population' not in api_data:
                return {"error": "API response is not in the expected format."}
            else:
                return api_data
        else:
            return {"error": "Failed to retrieve data from API"}
    except Exception as e:
        return {"error": str(e)}
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'file_a' not in request.files or 'file_b' not in request.files:
        return jsonify({
            "message": "Please upload a file for both DNA sequences.",
            "statusCode": 401
        })


    file_a = request.files['file_a']
    file_b = request.files['file_b']
    
    sequence_a = process_uploaded_file(file_a)
    sequence_b = process_uploaded_file(file_b)
    if file_a.filename == '' or file_b.filename == '':
        return jsonify({
            "message": "Please upload non-empty files for both DNA sequences.",
            "statusCode": 401
        })
    similarity = True

    # Load CSV files
    scoring_sys = ScoringSystems()
    scoring_sys.load_csv(SCORES_CSV)
    edit_cost_sys = ScoringSystems()
    edit_cost_sys.load_csv(EDIT_COST_CSV)

    analyzer = SequencesAnalyzer(sequence_a, sequence_b)

    if similarity:
       # alignment_a, alignment_b = analyzer.local_alignment()
        similarity_score = int(analyzer.similarity())  # Convert to integer
        match_status = "DNA MATCH" if similarity_score == 1000 else "DNA Not MATCH"
        similarity_percentage = int((similarity_score / 1000) * 100)  # Convert to integer
    else:
        pass
    return jsonify({
        #"alignment_a": alignment_a,
        #"alignment_b": alignment_b,
        "similarity_percentage": similarity_percentage, # 100
        #"similarity_score": similarity_score,
        "match_status": match_status ,  # "DNA MATCH" or "DNA Not MATCH"
        "message": "Successfull Comparison",
        "statusCode": 200
    })
    
@app.route('/result')
def result():
    return render_template('result.html')

#IDENTIFICATION
@app.route('/identify', methods=['POST'])

def identify():
    # Check if 'file' parameter is provided in the request
    if 'file' not in request.files:
        return jsonify({
            "message": "Please upload a file of DNA sequences.",
            "statusCode": 401 })
    # Get the uploaded file and selected status from the form
    file_a = request.files['file']
    if file_a.filename == '':
        return jsonify({
            "message": "Please upload non-empty files for both DNA sequences.",
            "statusCode": 401
        })
    similarity = True
    selected_status = request.form.get('status')

    # Retrieve API data
    api_data = retrieve_api_data(API_URL)
    if 'error' in api_data:
        return jsonify({
            "message": api_data['error'],
            "statusCode": 401 })

    # Retrieve DNA sequence from the uploaded file
    sequence_a = retrieve_dna_sequence_from_file(file_a)

    # Initialize variables for match information
    match_info = {}
    similarity_threshold = 2000  # Threshold for similarity score
    similarity_percentage = 0
    match_status = "No match found"

    # Extract necessary keys for match information
    info_keys = ['name', 'status', 'description', 'createdAt', 'updatedAt', 
                 'address', 'national_id', 'phone', 'gender', 'birthdate', 'bloodType']

    # Check if API data is a dictionary and contains 'population' key
    if isinstance(api_data, dict) and 'population' in api_data:
        # Filter API data based on selected status
        if selected_status == 'all':
            filtered_data = api_data['population']
        else:
            filtered_data = [entry for entry in api_data['population'] if entry.get('status') == selected_status]

        # Iterate over filtered data
        for entry in filtered_data:
            if 'DNA_sequence' in entry:
                # Retrieve DNA sequence for comparison
                sequence_b = entry['DNA_sequence'][:1000]
                # Calculate similarity score
                analyzer = SequencesAnalyzer(sequence_a, sequence_b)
                similarity_score = analyzer.similarity()
                # Check if similarity score meets the threshold
                if similarity_score == similarity_threshold:
                    # Extract match information
                    match_info = {key: entry[key] for key in info_keys if key in entry}
                    match_status = "DNA MATCH"
                    similarity_percentage = 100
                    break  # Exit loop if a match is found

    # Return the match information
    return jsonify({"match_info": match_info, 
                    "similarity_percentage": similarity_percentage,
                    "match_status": match_status ,
                    "message": "successful identification",
                    "statusCode": 200})

if __name__ == "__main__":
    app.run(debug=True)



# here i'm making the api (identify) case as 1000 from each one so total (2000) to match 
# Also with when i upload the two files case each one 500 letter so total (1000) to match