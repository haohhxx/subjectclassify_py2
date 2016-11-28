import re
s = 'as        sdasfsd'
result, number = re.subn('\s{2,10}','\t',s)
print result

