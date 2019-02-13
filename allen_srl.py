from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from contextlib import ExitStack
import argparse
import json
import os


class allenSRL: 
	def get_arguments():
	    parser = argparse.ArgumentParser()
	    
	    parser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')

	    parser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')	

	    parser.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')



	    args = parser.parse_args()

	    return args

	def get_predictor():
	   
	    return Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")


	def run(input_file,
		output_file,
		batch_size,
		print_to_console
		):

	    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz") 
	    def _run_predictor(batch_data):
	       
                if len(batch_data) == 1:
                    result = predictor.predict_json(batch_data[0])
		    
                    results = [result]
		     
		    
                else:
                    results = predictor.predict_batch_json(batch_data)
	
		
                return results

	       
	    batch_data = []
	    batch_result = []
	    print (input_file)
	   
	    for line in input_file:
                if not line.isspace():
                    line = {"sentence":line.strip()}
                    line = json.dumps(line)
                    json_data = predictor.load_line(line)
                    batch_data.append(json_data)
                    if len(batch_data) == batch_size:
                       res= _run_predictor(batch_data)
                       batch_result.append(res)
                       batch_data = []
	   
		    
	    if batch_data:
                res= _run_predictor(batch_data)
	
	    return batch_result
	   

	def main():
	    args = allenSRL.get_arguments()
	    predictor = allenSRL.get_predictor()
	    output_file = None
	    print_to_console = False

	    with ExitStack() as stack:
		
                input_file = stack.enter_context(args.input_file)           
                if args.output_file:
                    output_file = stack.enter_context(args.output_file)  

                if not args.output_file:
                    print_to_console = True
		
                result=allenSRL.run(input_file,output_file,args.batch_size,print_to_console)
		

                for output in result:
                    string_output = predictor.dump_line(output)
                    if print_to_console:
		       
                        print("prediction: ", string_output)
                    if output_file:
                        output_file.write(string_output)
	     
__end__ = '__end__'

if __name__ == '__main__':
    allenSRL.main()
    
 
