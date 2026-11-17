import json
from types import SimpleNamespace
import pytest
from pair import gsm8k_source as source
from pair.common import write_json,sha
from pair.evaluate import answer_number,aggregate


def no_hub(*a,**k):
    raise AssertionError('Local benchmark loading must not contact the Hub')


def test_offline_saved_ids_preserve_order_and_tokens(tmp_path,monkeypatch):
    monkeypatch.setattr(source,'load_dataset',no_hub)
    monkeypatch.setattr(source.AutoConfig,'from_pretrained',lambda *a,**k:SimpleNamespace(vocab_size=100))
    monkeypatch.setattr(source.AutoTokenizer,'from_pretrained',no_hub)
    rows=[{'input_ids':[4,3,2],'source_row':91},
          {'prompt_token_ids':[[7,8]],'source_row':5}, {'input_ids':[9,2]}]
    path=tmp_path/'gsm8k.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    before=path.read_bytes()
    result=source.prepare_gsm8k('local-target',tmp_path/'prepared.json',str(path),2)
    assert [r['input_ids'] for r in result['records']]==[[4,3,2],[7,8]]
    assert [r['source_index'] for r in result['records']]==[91,5]
    assert result['reference_answers_present']==0
    assert result['source_sha256']==sha(path) and path.read_bytes()==before
    prepared=tmp_path/'prepared.json'
    stamp=prepared.stat().st_mtime_ns
    again=source.prepare_gsm8k('local-target',prepared,str(path),2)
    assert again==result and prepared.stat().st_mtime_ns==stamp
    with pytest.raises(ValueError,match='Saved GSM8K prompts differ'):
        source.prepare_gsm8k('local-target',prepared,str(path),3)
    assert prepared.stat().st_mtime_ns==stamp


def test_local_text_formats_never_include_answers(tmp_path,monkeypatch):
    monkeypatch.setattr(source,'load_dataset',no_hub)
    monkeypatch.setattr(source.AutoConfig,'from_pretrained',lambda *a,**k:SimpleNamespace(vocab_size=100))
    calls=[]
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            calls.append(messages)
            assert kwargs['return_dict'] is False and kwargs['enable_thinking'] is False
            return [1,2]
        def encode(self,text,**kwargs):
            calls.append(text)
            assert kwargs['add_special_tokens'] is False
            return [3,4]
    monkeypatch.setattr(source.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    rows=[{'question':'Question one','answer':'secret derivation\n#### 12'},
          {'prompt':'Existing prompt'}, {'turns':['Existing turn']},
          {'messages':[{'role':'user','content':'Existing message'}]},
          {'text':'<|im_start|>user\nExisting chat'}]
    p=tmp_path/'prompts.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=source.prepare_gsm8k('target',tmp_path/'prepared.json',str(p))
    assert len(result['records'])==5 and result['reference_answers_present']==1
    assert 'secret derivation' not in str(calls)
    assert calls[0][0]['content']=='Question one'+source.MATH_SUFFIX
    assert calls[1][0]['content']=='Existing prompt'
    assert result['records'][-1]['input_ids']==[3,4]


@pytest.mark.parametrize('row',[
    {'input_ids':[-1,2]}, {'input_ids':[100,1]}, {'input_ids':[1.2,3]},
    {'input_ids':[[1,2],[3,4]]}, {'input_ids':[]},
    {'input_ids':[1,2],'attention_mask':[1,0]},
    {'input_ids':[1,2],'loss_mask':[0,1]},
])
def test_invalid_saved_prompts_rejected(row):
    with pytest.raises(ValueError): source.token_ids(row,100)


@pytest.mark.parametrize('correct,expected,status',[
    ([None,None],None,'unavailable_no_parseable_references'),
    ([True,None],1.0,'partial'), ([True,False],0.5,'available')])
def test_missing_answers_do_not_create_fake_accuracy(tmp_path,correct,expected,status):
    rows=[{'eval_index':i,'rank':0,'acceptance_lengths':[3+i*2], 'seconds':1.0,
           'correct':c,'predicted_answer':'12','truncated':False,'output_ids':[1,2],
           'greedy_parity_checked':True} for i,c in enumerate(correct)]
    write_json(tmp_path/'rank-0.json',{'records':rows})
    result=aggregate(tmp_path,1,2)
    assert result['gsm8k_numeric_exact_match']==expected
    assert result['accuracy_status']==status
    assert result['acceptance_length_including_bonus']==4


def test_gold_formats_and_missing_reference():
    assert answer_number(None,gold=True) is None
    assert answer_number('work\n#### 12',gold=True)==answer_number('12',gold=True)
    assert answer_number(12,gold=True)==answer_number('\\boxed{12}',gold=True)
    assert answer_number('unparseable reference',gold=True) is None
