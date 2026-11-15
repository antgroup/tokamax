"""Pinned main/PR benchmark manifest. Case parameters are public API inputs."""
def case(b,t,h,n,*,norm=True,fixed=False,cp=1,remat=False,over=False,preactivated=False,state=False):
    return dict(b=b,t=t,h=h,n=n,norm=norm,fixed=fixed,cp=cp,remat=remat,over=over,preactivated=preactivated,state=state)
CASES={
 'forward':{
  'arg_fixed':case(1,8192,32,1,fixed=True),
  'arg_packed':case(1,8192,32,25),
  'compact_packed':case(1,8192,32,25,norm=False),
  'fragmented':case(1,2048,8,512,norm=False),
  'long64k':case(1,65536,8,64,norm=False),
  'batch2':case(2,8192,64,25,norm=False),
 },
 'backward':{
  'arg_fixed':case(1,8192,32,1,fixed=True),
  'arg_packed':case(1,8192,32,25),
  'compact_packed':case(1,8192,32,25,norm=False),
  'fragmented':case(1,2048,8,512,norm=False),
  'remat_packed':case(1,8192,32,25,remat=True),
 },
 'cp':{
  'cp4_normal':case(1,512,8,32,cp=4),
  'cp8_normal':case(2,1024,8,32,cp=8),
  'cp4_remat':case(1,512,8,32,cp=4,remat=True),
  'cp8_capacity':case(2,1024,4,8192,cp=8,norm=False),
  'cp8_overflow':case(2,1024,4,8192,cp=8,norm=False,over=True),
  'cp4_mega':case(2,512,8,2,cp=4,norm=False,preactivated=True),
 },
 'inference':{
  'tokens512':case(1,512,8,2),
  'tokens8192':case(1,8192,8,2),
  'arg_packed':case(1,8192,32,25),
  'fragmented':case(1,2048,8,512),
  'state512':case(1,512,8,2,state=True),
 },
}
VARIANTS={'forward':['main','default','aligned','compact'],
          'backward':['main','default','forward','optimized','compact'],
          'cp':['main','default','fused','workspace','mega','parent'],
          'inference':['main','default','native']}
def selected(suite,variant,name):
    if suite=='backward' and variant=='compact':return name in ('compact_packed','fragmented')
    if suite=='cp' and variant=='mega':return name=='cp4_mega'
    return True
def profile_case(suite,variant,name,mode):
    if suite=='forward':return (name=='arg_packed' and variant in ('main','aligned')) or (name=='compact_packed' and variant in ('main','compact'))
    if suite=='backward':return name=='compact_packed' and variant in ('main','compact')
    if suite=='cp':return (name=='cp8_normal' and variant in ('main','fused','workspace','parent')) or (name=='cp8_capacity' and variant in ('main','workspace','parent')) or (name=='cp4_mega' and variant in ('main','mega'))
    return name=='tokens8192' and variant in ('main','native')
