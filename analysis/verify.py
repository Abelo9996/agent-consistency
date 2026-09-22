"""Re-derive every headline number the revision touches, from the released traces."""
import json, numpy as np
from itertools import combinations
from scipy import stats

d = json.load(open('results/experiment_20260705_205523.json'))
tss = np.array([c['metrics']['tool_sequence_similarity'] for c in d])
ac  = np.array([c['metrics']['argument_consistency'] for c in d])
om  = np.array([c['metrics']['output_agreement']['exact_match_rate'] for c in d])
cat = np.array([c['category'] for c in d])
mod = np.array([c['model'] for c in d])

print('n conditions', len(d), 'n traces', sum(len(c['traces']) for c in d))
print('TSS mean %.4f' % tss.mean(), 'AC mean %.4f' % ac.mean())
def ci(x):
    se = x.std(ddof=1)/np.sqrt(len(x)); return x.mean()-1.96*se, x.mean()+1.96*se
print('TSS 95%% CI [%.3f, %.3f]' % ci(tss), ' AC 95%% CI [%.3f, %.3f]' % ci(ac))
t,p = stats.ttest_rel(tss, ac)
diff = tss-ac
dz = diff.mean()/diff.std(ddof=1)
sp = np.sqrt((tss.var(ddof=1)+ac.var(ddof=1))/2)
print('paired t=%.3f p=%.3e  Cohen dz=%.3f  pooled-sd d=%.3f' % (t,p,dz,diff.mean()/sp))

print()
print('Output exact match (pairwise, full final_response string, .strip()):')
print('  mean over conditions %.4f' % om.mean())
# pooled over all pairs
tot=match=0
for c in d:
    resp=[t_['final_response'].strip() for t_ in c['traces'] if t_.get('final_response')]
    for a,b in combinations(resp,2):
        tot+=1; match += (a==b)
print('  pooled over all response pairs: %d/%d = %.4f' % (match,tot,match/tot))

print()
print('Ambiguity (category == ambiguous) vs rest, AC:')
amb = ac[cat=='ambiguous']; rest = ac[cat!='ambiguous']
print('  amb n=%d mean %.4f | rest n=%d mean %.4f | rel drop %.1f%%' % (len(amb),amb.mean(),len(rest),rest.mean(),100*(1-amb.mean()/rest.mean())))
t2,p2 = stats.ttest_ind(amb,rest,equal_var=False)
spd = np.sqrt(((len(amb)-1)*amb.var(ddof=1)+(len(rest)-1)*rest.var(ddof=1))/(len(amb)+len(rest)-2))
print('  Welch t=%.3f p=%.4f  Cohen d=%.3f' % (t2,p2,(rest.mean()-amb.mean())/spd))

print()
for name,x in (('TSS',tss),('AC',ac)):
    groups=[x[mod==m] for m in np.unique(mod)]
    F,pv = stats.f_oneway(*groups)
    ssb = sum(len(g)*(g.mean()-x.mean())**2 for g in groups)
    print('  ANOVA model on %s: F=%.3f p=%.4f eta2=%.3f' % (name,F,pv,ssb/((x-x.mean())**2).sum()))

print()
print('Per-model TSS/AC/uniq:')
for m in np.unique(mod):
    s=mod==m
    u=np.array([c['metrics']['unique_sequences'] for c in d])[s]
    print('  %-22s TSS %.3f  AC %.3f  uniq %.2f  n=%d' % (m,tss[s].mean(),ac[s].mean(),u.mean(),s.sum()))

print()
print('Trace length (number of tool calls) distribution:')
L=[]
for c in d:
    for t_ in c['traces']:
        L.append((c['model'],c['task_id'],c['category'],len(t_.get('tool_calls') or [])))
arr=np.array([x[3] for x in L])
print('  n traces %d  mean %.2f  sd %.2f  median %.1f  min %d max %d' % (len(arr),arr.mean(),arr.std(ddof=1),np.median(arr),arr.min(),arr.max()))
import collections
print('  histogram:',sorted(collections.Counter(arr.tolist()).items()))
print('  by model:')
for m in np.unique([x[0] for x in L]):
    a=np.array([x[3] for x in L if x[0]==m])
    print('    %-22s mean %.2f sd %.2f median %.1f range %d-%d' % (m,a.mean(),a.std(ddof=1),np.median(a),a.min(),a.max()))
print('  by category:')
for cc in ['retrieval','scheduling','computation','composition','ambiguous']:
    a=np.array([x[3] for x in L if x[2]==cc])
    if len(a): print('    %-14s mean %.2f sd %.2f median %.1f range %d-%d' % (cc,a.mean(),a.std(ddof=1),np.median(a),a.min(),a.max()))
