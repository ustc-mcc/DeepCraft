#coding=utf-8 

from django.utils.html import escape
from django.shortcuts import render
#from forms import LoginForm, RegisterForm, ListenForm
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect, get_object_or_404
#from .models import NewUser, Notes
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile

from audio_api import run_text2audio

def index(request):
    return render(request, 'index.html')

def record(request):
    return render(request, 'record_audio.html')

def sr(request):
    return render(request, 'sr.html')

def play(request, page):
	render_dict = {'page': page}
	return render(request, 'play.html', render_dict)


def pron(request):	
	f = open('vis/result.wav','r')
	result = f.readline()
	print result
	
@csrf_exempt 
def upload_img(request):
	if request.method == "POST":
		print 'FILES:', request.FILES
		print 'body:', request.body
		print 'type body:', type(request.body)
		print 'len body:', len(request.body)
		f = open('tmp.jpg', 'wb')
#		f.write(image_url.decode('base64'))
		
#	return HttpResponse("ok")	
	return HttpResponse(escape(repr(request)))

from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.template import RequestContext
import subprocess
import json 
import os
import sys
import math

sys.path.insert(0, '../im2txt/')

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "../im2txt/data/model/model.ckpt-1000000", "Model checkpoint file or directory containing a model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "../im2txt/data/mscoco/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "", "File pattern or comma-separated list of file patterns of image files.")
tf.logging.set_verbosity(tf.logging.INFO)

#g = tf.Graph()
#g.as_default()
#model = inference_wrapper.InferenceWrapper()
#restore_fn = model.build_graph_from_config(configuration.ModelConfig(), FLAGS.checkpoint_path)
#g.finalize()
vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
# sess = tf.Session(graph=g)
# restore_fn(sess)
# generator = caption_generator.CaptionGenerator(model, vocab)

def load_sentence(json_path):
    sentences = []
    with open(json_path) as f_json:
        for line in f_json:
	    sentences.append(line)
    return sentences

def parse_json(json_path):
    sentences = []
    json_file = open(json_path) 
    for line in json_file:
        json_obj = json.loads(line)
        for i in range(len(json_obj)):
            sentences.append(json_obj[i]['caption'])
            
    return sentences
    
#def index(request):
#	return render_to_response('index.html')	

def caption(request):
    sentences = "" 
    imgs_saved_path = 'VideoSR/static/imgs/hackxsjtu/'
    json_path = 'vis/vis.json'
    try: 
        os.remove(json_path)
    except:
        pass
#    try:
#    	subprocess.call(['rm', imgs_saved_path+'*'])
#    except:
#    	pass

    #upload file to images saved directory.
    if request.method == "POST":
        f = request.FILES['tmp_image'] 
        with open(imgs_saved_path + 'tmp.jpg', 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)
    
        if True:
	    g = tf.Graph()
	    with g.as_default():
	        model = inference_wrapper.InferenceWrapper()
		restore_fn = model.build_graph_from_config(configuration.ModelConfig(), FLAGS.checkpoint_path)
	    g.finalize()
	    sess = tf.Session(graph=g)
	    restore_fn(sess)
	    generator = caption_generator.CaptionGenerator(model, vocab)
            imgname = imgs_saved_path + 'tmp.jpg'
	    filenames = []
	    print "HERE"
	    for file_pattern in imgname.split(","):
	        filenames.extend(tf.gfile.Glob(file_pattern))
	    tf.logging.info("Running caption generation on %d files matching %s", len(filenames), imgname)
	   # generator = caption_generator.CaptionGenerator(model, vocab)
	    jsfile = 'vis/vis.json'
	    f_json = open(jsfile, 'w')
	    for filename in filenames:
	        with tf.gfile.GFile(filename, "r") as f:
		    image = f.read()
	        captions = generator.beam_search(sess, image)
		print "Captions for image %s: " % os.path.basename(filename)
		ss = []
		for i, caption in enumerate(captions):
		    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
		    sentence = " ".join(sentence)
		    ss.append(sentence)
		    print(" %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
	    f_json.write(ss[0])
	    run_text2audio(ss[0])
	    f_json.close()
	    result = 0

#            result = subprocess.call(['bash', '/home/nimbix/im2txt/test/run.sh']) 
        else:
            a = sys.exc_info()[0]
            assert False
        if result == 0:
            print('all_ok')
            #sentences = parse_json(json_path) 
	    sentences = load_sentence(json_path)
        else:
            assert False
    
    return render_to_response('caption.html', {'sentences':sentences}, context_instance=RequestContext(request))
    
    
    #parse json file,get stenences
    

