Ń
Ů&ź&
9
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type"
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

A
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype

Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
7
Less
x"T
y"T
z
"
Ttype:
2		
!
LoopCond	
input


output

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sigmoid
x"T
y"T"
Ttype:	
2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
,
Tanh
x"T
y"T"
Ttype:	
2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
¸
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.3.02
b'unknown'ç

global_step/Initializer/zerosConst*
value	B	 R *
_output_shapes
: *
_class
loc:@global_step*
dtype0	

global_step
VariableV2*
shared_name *
shape: *
_output_shapes
: *
	container *
_class
loc:@global_step*
dtype0	
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
validate_shape(*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
f
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
T
	map/ShapeShapePlaceholder*
T0*
_output_shapes
:*
out_type0
a
map/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
c
map/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
c
map/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

map/strided_sliceStridedSlice	map/Shapemap/strided_slice/stackmap/strided_slice/stack_1map/strided_slice/stack_2*
Index0*
_output_shapes
: *
new_axis_mask *

begin_mask *
ellipsis_mask *
T0*
shrink_axis_mask*
end_mask 
ş
map/TensorArrayTensorArrayV3map/strided_slice*
element_shape:*
tensor_array_name *
clear_after_read(*
dynamic_size( *
_output_shapes

:: *
dtype0
g
map/TensorArrayUnstack/ShapeShapePlaceholder*
T0*
_output_shapes
:*
out_type0
t
*map/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
v
,map/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
v
,map/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ě
$map/TensorArrayUnstack/strided_sliceStridedSlicemap/TensorArrayUnstack/Shape*map/TensorArrayUnstack/strided_slice/stack,map/TensorArrayUnstack/strided_slice/stack_1,map/TensorArrayUnstack/strided_slice/stack_2*
Index0*
_output_shapes
: *
new_axis_mask *

begin_mask *
ellipsis_mask *
T0*
shrink_axis_mask*
end_mask 
d
"map/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"map/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Ä
map/TensorArrayUnstack/rangeRange"map/TensorArrayUnstack/range/start$map/TensorArrayUnstack/strided_slice"map/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map/TensorArraymap/TensorArrayUnstack/rangePlaceholdermap/TensorArray:1*
T0*
_output_shapes
: *
_class
loc:@Placeholder
K
	map/ConstConst*
value	B : *
_output_shapes
: *
dtype0
ź
map/TensorArray_1TensorArrayV3map/strided_slice*
element_shape:*
tensor_array_name *
clear_after_read(*
dynamic_size( *
_output_shapes

:: *
dtype0

map/while/EnterEnter	map/Const*
parallel_iterations
*
T0*
_output_shapes
: *$

frame_namemap/while/map/while/*
is_constant( 
¤
map/while/Enter_1Entermap/TensorArray_1:1*
parallel_iterations
*
T0*
_output_shapes
: *$

frame_namemap/while/map/while/*
is_constant( 
n
map/while/MergeMergemap/while/Entermap/while/NextIteration*
T0*
_output_shapes
: : *
N
t
map/while/Merge_1Mergemap/while/Enter_1map/while/NextIteration_1*
T0*
_output_shapes
: : *
N
Ľ
map/while/Less/EnterEntermap/strided_slice*
parallel_iterations
*
T0*
_output_shapes
: *$

frame_namemap/while/map/while/*
is_constant(
^
map/while/LessLessmap/while/Mergemap/while/Less/Enter*
T0*
_output_shapes
: 
F
map/while/LoopCondLoopCondmap/while/Less*
_output_shapes
: 

map/while/SwitchSwitchmap/while/Mergemap/while/LoopCond*
T0*
_output_shapes
: : *"
_class
loc:@map/while/Merge

map/while/Switch_1Switchmap/while/Merge_1map/while/LoopCond*
T0*
_output_shapes
: : *$
_class
loc:@map/while/Merge_1
S
map/while/IdentityIdentitymap/while/Switch:1*
T0*
_output_shapes
: 
W
map/while/Identity_1Identitymap/while/Switch_1:1*
T0*
_output_shapes
: 
´
!map/while/TensorArrayReadV3/EnterEntermap/TensorArray*
parallel_iterations
*
T0*
_output_shapes
:*$

frame_namemap/while/map/while/*
is_constant(
á
#map/while/TensorArrayReadV3/Enter_1Enter>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations
*
T0*
_output_shapes
: *$

frame_namemap/while/map/while/*
is_constant(
ą
map/while/TensorArrayReadV3TensorArrayReadV3!map/while/TensorArrayReadV3/Entermap/while/Identity#map/while/TensorArrayReadV3/Enter_1*
_output_shapes
: *
dtype0

map/while/PyFuncPyFuncmap/while/TensorArrayReadV3*
_output_shapes
:*
Tout
2*
Tin
2*
token
pyfunc_2
í
3map/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap/TensorArray_1*
_output_shapes
:*$

frame_namemap/while/map/while/*
parallel_iterations
*
T0*#
_class
loc:@map/while/PyFunc*
is_constant(
ú
-map/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33map/while/TensorArrayWrite/TensorArrayWriteV3/Entermap/while/Identitymap/while/PyFuncmap/while/Identity_1*
T0*
_output_shapes
: *#
_class
loc:@map/while/PyFunc
f
map/while/add/yConst^map/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Z
map/while/addAddmap/while/Identitymap/while/add/y*
T0*
_output_shapes
: 
X
map/while/NextIterationNextIterationmap/while/add*
T0*
_output_shapes
: 
z
map/while/NextIteration_1NextIteration-map/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
I
map/while/ExitExitmap/while/Switch*
T0*
_output_shapes
: 
M
map/while/Exit_1Exitmap/while/Switch_1*
T0*
_output_shapes
: 

&map/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map/TensorArray_1map/while/Exit_1*
_output_shapes
: *$
_class
loc:@map/TensorArray_1

 map/TensorArrayStack/range/startConst*
value	B : *
_output_shapes
: *
dtype0*$
_class
loc:@map/TensorArray_1

 map/TensorArrayStack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0*$
_class
loc:@map/TensorArray_1
ć
map/TensorArrayStack/rangeRange map/TensorArrayStack/range/start&map/TensorArrayStack/TensorArraySizeV3 map/TensorArrayStack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_class
loc:@map/TensorArray_1
ä
(map/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_1map/TensorArrayStack/rangemap/while/Exit_1*
element_shape:*
_output_shapes
:*
dtype0*$
_class
loc:@map/TensorArray_1
g
SizeSize(map/TensorArrayStack/TensorArrayGatherV3*
T0*
_output_shapes
: *
out_type0
Q
Reshape/shape/0Const*
value	B :*
_output_shapes
: *
dtype0
f
Reshape/shapePackReshape/shape/0Size*

axis *
T0*
_output_shapes
:*
N

ReshapeReshape(map/TensorArrayStack/TensorArrayGatherV3Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Y
Reshape_1/shapeConst*
valueB:*
_output_shapes
:*
dtype0
^
	Reshape_1ReshapeSizeReshape_1/shape*
T0*
_output_shapes
:*
Tshape0
o
embedding/random_uniform/shapeConst*
valueB"@     *
_output_shapes
:*
dtype0
a
embedding/random_uniform/minConst*
valueB
 *  ż*
_output_shapes
: *
dtype0
a
embedding/random_uniform/maxConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
§
&embedding/random_uniform/RandomUniformRandomUniformembedding/random_uniform/shape*

seed*
T0*
seed2;*
dtype0*
_output_shapes
:	Ŕ

embedding/random_uniform/subSubembedding/random_uniform/maxembedding/random_uniform/min*
T0*
_output_shapes
: 

embedding/random_uniform/mulMul&embedding/random_uniform/RandomUniformembedding/random_uniform/sub*
T0*
_output_shapes
:	Ŕ

embedding/random_uniformAddembedding/random_uniform/mulembedding/random_uniform/min*
T0*
_output_shapes
:	Ŕ

embedding/W
VariableV2*
shared_name *
shape:	Ŕ*
_output_shapes
:	Ŕ*
dtype0*
	container 
ś
embedding/W/AssignAssignembedding/Wembedding/random_uniform*
use_locking(*
T0*
validate_shape(*
_class
loc:@embedding/W*
_output_shapes
:	Ŕ
s
embedding/W/readIdentityembedding/W*
T0*
_output_shapes
:	Ŕ*
_class
loc:@embedding/W
Ă
embedding/embedding_lookupGatherembedding/W/readReshape*
validate_indices(*
Tparams0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@embedding/W*
Tindices0
]
DropoutWrapperInit/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
_
DropoutWrapperInit/Const_1Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
\
DropoutWrapperInit/Const_2Const*
value	B :*
_output_shapes
: *
dtype0
_
DropoutWrapperInit_1/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
a
DropoutWrapperInit_1/Const_1Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
^
DropoutWrapperInit_1/Const_2Const*
value	B :*
_output_shapes
: *
dtype0
F
RankConst*
value	B :*
_output_shapes
: *
dtype0
M
range/startConst*
value	B :*
_output_shapes
: *
dtype0
M
range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
V
rangeRangerange/startRankrange/delta*

Tidx0*
_output_shapes
:
`
concat/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
M
concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
q
concatConcatV2concat/values_0rangeconcat/axis*
T0*

Tidx0*
_output_shapes
:*
N
}
	transpose	Transposeembedding/embedding_lookupconcat*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tperm0
K
sequence_lengthIdentity	Reshape_1*
T0*
_output_shapes
:

Irnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:*
_output_shapes
:*
dtype0

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:d*
_output_shapes
:*
dtype0

Ornn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ů
Jrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Irnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ConstKrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Ornn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
_output_shapes
:*
dtype0

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:d*
_output_shapes
:*
dtype0

Ornn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

Irnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillJrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concatOrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:d

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
_output_shapes
:*
dtype0

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_5Const*
valueB:d*
_output_shapes
:*
dtype0

Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
˙
Lrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_4Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_5Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
_output_shapes
:*
dtype0

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_7Const*
valueB:d*
_output_shapes
:*
dtype0

Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillLrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:d

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ConstConst*
valueB:*
_output_shapes
:*
dtype0

Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_1Const*
valueB:d*
_output_shapes
:*
dtype0

Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0

Lrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concatConcatV2Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/ConstMrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_1Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N

Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_2Const*
valueB:*
_output_shapes
:*
dtype0

Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_3Const*
valueB:d*
_output_shapes
:*
dtype0

Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zerosFillLrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concatQrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:d

Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_4Const*
valueB:*
_output_shapes
:*
dtype0

Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_5Const*
valueB:d*
_output_shapes
:*
dtype0

Srnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0

Nrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1ConcatV2Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_4Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_5Srnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N

Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_6Const*
valueB:*
_output_shapes
:*
dtype0

Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/Const_7Const*
valueB:d*
_output_shapes
:*
dtype0

Srnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
Ł
Mrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1FillNrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/concat_1Srnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:d
S
	rnn/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
S
	rnn/stackConst*
valueB:*
_output_shapes
:*
dtype0
M
	rnn/EqualEqual	rnn/Shape	rnn/stack*
T0*
_output_shapes
:
S
	rnn/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Y
rnn/AllAll	rnn/Equal	rnn/Const*
_output_shapes
: *

Tidx0*
	keep_dims( 

rnn/Assert/ConstConst*@
value7B5 B/Expected shape for Tensor sequence_length:0 is *
_output_shapes
: *
dtype0
c
rnn/Assert/Const_1Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0

rnn/Assert/Assert/data_0Const*@
value7B5 B/Expected shape for Tensor sequence_length:0 is *
_output_shapes
: *
dtype0
i
rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0

rnn/Assert/AssertAssertrnn/Allrnn/Assert/Assert/data_0	rnn/stackrnn/Assert/Assert/data_2	rnn/Shape*
	summarize*
T
2
e
rnn/CheckSeqLenIdentitysequence_length^rnn/Assert/Assert*
T0*
_output_shapes
:
T
rnn/Shape_1Shape	transpose*
T0*
_output_shapes
:*
out_type0
a
rnn/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
c
rnn/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
c
rnn/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

rnn/strided_sliceStridedSlicernn/Shape_1rnn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*
_output_shapes
: *
new_axis_mask *

begin_mask *
ellipsis_mask *
T0*
shrink_axis_mask*
end_mask 
U
rnn/Const_1Const*
valueB:*
_output_shapes
:*
dtype0
U
rnn/Const_2Const*
valueB:d*
_output_shapes
:*
dtype0
Q
rnn/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
{

rnn/concatConcatV2rnn/Const_1rnn/Const_2rnn/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
T
rnn/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
W
	rnn/zerosFill
rnn/concatrnn/zeros/Const*
T0*
_output_shapes

:d
U
rnn/Const_3Const*
valueB: *
_output_shapes
:*
dtype0
j
rnn/MinMinrnn/CheckSeqLenrnn/Const_3*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
U
rnn/Const_4Const*
valueB: *
_output_shapes
:*
dtype0
j
rnn/MaxMaxrnn/CheckSeqLenrnn/Const_4*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
J
rnn/timeConst*
value	B : *
_output_shapes
: *
dtype0
Ň
rnn/TensorArrayTensorArrayV3rnn/strided_slice*
element_shape:*/
tensor_array_namernn/dynamic_rnn/output_0*
clear_after_read(*
dynamic_size( *
_output_shapes

:: *
dtype0
Ó
rnn/TensorArray_1TensorArrayV3rnn/strided_slice*
element_shape:*.
tensor_array_namernn/dynamic_rnn/input_0*
clear_after_read(*
dynamic_size( *
_output_shapes

:: *
dtype0
e
rnn/TensorArrayUnstack/ShapeShape	transpose*
T0*
_output_shapes
:*
out_type0
t
*rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ě
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
_output_shapes
: *
new_axis_mask *

begin_mask *
ellipsis_mask *
T0*
shrink_axis_mask*
end_mask 
d
"rnn/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Ä
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	transposernn/TensorArray_1:1*
T0*
_output_shapes
: *
_class
loc:@transpose

rnn/while/EnterEnterrnn/time*
parallel_iterations *
T0*
_output_shapes
: *$

frame_namernn/while/rnn/while/*
is_constant( 
˘
rnn/while/Enter_1Enterrnn/TensorArray:1*
parallel_iterations *
T0*
_output_shapes
: *$

frame_namernn/while/rnn/while/*
is_constant( 
â
rnn/while/Enter_2EnterIrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
parallel_iterations *
T0*
_output_shapes

:d*$

frame_namernn/while/rnn/while/*
is_constant( 
ä
rnn/while/Enter_3EnterKrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
parallel_iterations *
T0*
_output_shapes

:d*$

frame_namernn/while/rnn/while/*
is_constant( 
ä
rnn/while/Enter_4EnterKrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros*
parallel_iterations *
T0*
_output_shapes

:d*$

frame_namernn/while/rnn/while/*
is_constant( 
ć
rnn/while/Enter_5EnterMrnn/MultiRNNCellZeroState/DropoutWrapperZeroState_1/LSTMCellZeroState/zeros_1*
parallel_iterations *
T0*
_output_shapes

:d*$

frame_namernn/while/rnn/while/*
is_constant( 
n
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
T0*
_output_shapes
: : *
N
t
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
T0*
_output_shapes
: : *
N
|
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0* 
_output_shapes
:d: *
N
|
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0* 
_output_shapes
:d: *
N
|
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4*
T0* 
_output_shapes
:d: *
N
|
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5*
T0* 
_output_shapes
:d: *
N
Ľ
rnn/while/Less/EnterEnterrnn/strided_slice*
parallel_iterations *
T0*
_output_shapes
: *$

frame_namernn/while/rnn/while/*
is_constant(
^
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0*
_output_shapes
: 
F
rnn/while/LoopCondLoopCondrnn/while/Less*
_output_shapes
: 

rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
T0*
_output_shapes
: : *"
_class
loc:@rnn/while/Merge

rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
T0*
_output_shapes
: : *$
_class
loc:@rnn/while/Merge_1

rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*
T0*(
_output_shapes
:d:d*$
_class
loc:@rnn/while/Merge_2

rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*
T0*(
_output_shapes
:d:d*$
_class
loc:@rnn/while/Merge_3

rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*
T0*(
_output_shapes
:d:d*$
_class
loc:@rnn/while/Merge_4

rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*
T0*(
_output_shapes
:d:d*$
_class
loc:@rnn/while/Merge_5
S
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0*
_output_shapes
: 
W
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0*
_output_shapes
: 
_
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0*
_output_shapes

:d
_
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0*
_output_shapes

:d
_
rnn/while/Identity_4Identityrnn/while/Switch_4:1*
T0*
_output_shapes

:d
_
rnn/while/Identity_5Identityrnn/while/Switch_5:1*
T0*
_output_shapes

:d
ś
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
parallel_iterations *
T0*
_output_shapes
:*$

frame_namernn/while/rnn/while/*
is_constant(
á
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
T0*
_output_shapes
: *$

frame_namernn/while/rnn/while/*
is_constant(
š
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity#rnn/while/TensorArrayReadV3/Enter_1*
_output_shapes

:*
dtype0
Ű
Krnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"     *
_output_shapes
:*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0
Í
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *˝çŮ˝*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0
Í
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *˝çŮ=*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0
Â
Srnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformKrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shape*

seed*
seed2š*
dtype0* 
_output_shapes
:
*
T0*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Ć
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/subSubIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Ú
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMulSrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Ě
Ernn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniformAddIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
á
*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
VariableV2*
shared_name *
shape:
* 
_output_shapes
:
*
	container *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0
Á
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AssignAssign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelErnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:


/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/readIdentity*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0* 
_output_shapes
:

Ť
Trnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ł
Ornn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_3Trnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/concat/axis*
T0*

Tidx0*
_output_shapes
:	*
N

Urnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
parallel_iterations *
T0* 
_output_shapes
:
*$

frame_namernn/while/rnn/while/*
is_constant(
Ń
Ornn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/MatMulMatMulOrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/concatUrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/MatMul/Enter*
transpose_a( *
T0*
_output_shapes
:	*
transpose_b( 
Ć
:rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/ConstConst*
valueB*    *
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
dtype0
Ó
(rnn/multi_rnn_cell/cell_0/lstm_cell/bias
VariableV2*
shared_name *
shape:*
_output_shapes	
:*
	container *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
dtype0
Ť
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AssignAssign(rnn/multi_rnn_cell/cell_0/lstm_cell/bias:rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/Const*
use_locking(*
T0*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:

-rnn/multi_rnn_cell/cell_0/lstm_cell/bias/readIdentity(rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
T0*
_output_shapes	
:

Vrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/BiasAdd/EnterEnter-rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read*
parallel_iterations *
T0*
_output_shapes	
:*$

frame_namernn/while/rnn/while/*
is_constant(
Ĺ
Prnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/BiasAddBiasAddOrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/MatMulVrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/BiasAdd/Enter*
T0*
_output_shapes
:	*
data_formatNHWC

Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/ConstConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ľ
Nrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ç
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/splitSplitNrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split/split_dimPrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/BiasAdd*
	num_split*
T0*<
_output_shapes*
(:d:d:d:d

Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add/yConst^rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0

Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/addAddFrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split:2Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add/y*
T0*
_output_shapes

:d
ž
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/SigmoidSigmoidBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add*
T0*
_output_shapes

:d
Đ
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mulMulFrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoidrnn/while/Identity_2*
T0*
_output_shapes

:d
Â
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoid_1SigmoidDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split*
T0*
_output_shapes

:d
ź
Crnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/TanhTanhFrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split:1*
T0*
_output_shapes

:d

Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_1MulHrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoid_1Crnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Tanh*
T0*
_output_shapes

:d
ţ
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add_1AddBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mulDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_1*
T0*
_output_shapes

:d
Ä
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoid_2SigmoidFrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split:3*
T0*
_output_shapes

:d
ź
Ernn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Tanh_1TanhDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add_1*
T0*
_output_shapes

:d

Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_2MulHrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoid_2Ernn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Tanh_1*
T0*
_output_shapes

:d
Ű
Krnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"Č     *
_output_shapes
:*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtype0
Í
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *ÍĚĚ˝*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtype0
Í
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÍĚĚ=*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtype0
Â
Srnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformKrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shape*

seed*
seed2Ű*
dtype0* 
_output_shapes
:
Č*
T0*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Ć
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/subSubIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Ú
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulMulSrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
Č*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Ě
Ernn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniformAddIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
Č*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
á
*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
VariableV2*
shared_name *
shape:
Č* 
_output_shapes
:
Č*
	container *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtype0
Á
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AssignAssign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelErnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel* 
_output_shapes
:
Č

/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/readIdentity*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
T0* 
_output_shapes
:
Č
Ť
Trnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ě
Ornn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/concatConcatV2Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_2rnn/while/Identity_5Trnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/concat/axis*
T0*

Tidx0*
_output_shapes
:	Č*
N

Urnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
parallel_iterations *
T0* 
_output_shapes
:
Č*$

frame_namernn/while/rnn/while/*
is_constant(
Ń
Ornn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/MatMulMatMulOrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/concatUrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/MatMul/Enter*
transpose_a( *
T0*
_output_shapes
:	*
transpose_b( 
Ć
:rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/ConstConst*
valueB*    *
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
dtype0
Ó
(rnn/multi_rnn_cell/cell_1/lstm_cell/bias
VariableV2*
shared_name *
shape:*
_output_shapes	
:*
	container *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
dtype0
Ť
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AssignAssign(rnn/multi_rnn_cell/cell_1/lstm_cell/bias:rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/Const*
use_locking(*
T0*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes	
:

-rnn/multi_rnn_cell/cell_1/lstm_cell/bias/readIdentity(rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
T0*
_output_shapes	
:

Vrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/BiasAdd/EnterEnter-rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read*
parallel_iterations *
T0*
_output_shapes	
:*$

frame_namernn/while/rnn/while/*
is_constant(
Ĺ
Prnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/BiasAddBiasAddOrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/MatMulVrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/BiasAdd/Enter*
T0*
_output_shapes
:	*
data_formatNHWC

Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/ConstConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ľ
Nrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ç
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/splitSplitNrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split/split_dimPrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/BiasAdd*
	num_split*
T0*<
_output_shapes*
(:d:d:d:d

Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add/yConst^rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0

Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/addAddFrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split:2Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add/y*
T0*
_output_shapes

:d
ž
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/SigmoidSigmoidBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add*
T0*
_output_shapes

:d
Đ
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mulMulFrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoidrnn/while/Identity_4*
T0*
_output_shapes

:d
Â
Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoid_1SigmoidDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split*
T0*
_output_shapes

:d
ź
Crnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/TanhTanhFrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split:1*
T0*
_output_shapes

:d

Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_1MulHrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoid_1Crnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Tanh*
T0*
_output_shapes

:d
ţ
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add_1AddBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mulDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_1*
T0*
_output_shapes

:d
Ä
Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoid_2SigmoidFrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split:3*
T0*
_output_shapes

:d
ź
Ernn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Tanh_1TanhDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add_1*
T0*
_output_shapes

:d

Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2MulHrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoid_2Ernn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Tanh_1*
T0*
_output_shapes

:d
Ż
rnn/while/GreaterEqual/EnterEnterrnn/CheckSeqLen*
parallel_iterations *
T0*
_output_shapes
:*$

frame_namernn/while/rnn/while/*
is_constant(
}
rnn/while/GreaterEqualGreaterEqualrnn/while/Identityrnn/while/GreaterEqual/Enter*
T0*
_output_shapes
:

rnn/while/Select/EnterEnter	rnn/zeros*
_output_shapes

:d*$

frame_namernn/while/rnn/while/*
parallel_iterations *
T0*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2*
is_constant(

rnn/while/SelectSelectrnn/while/GreaterEqualrnn/while/Select/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2*
T0*
_output_shapes

:d*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2

rnn/while/GreaterEqual_1GreaterEqualrnn/while/Identityrnn/while/GreaterEqual/Enter*
T0*
_output_shapes
:

rnn/while/Select_1Selectrnn/while/GreaterEqual_1rnn/while/Identity_2Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add_1*
T0*
_output_shapes

:d*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add_1

rnn/while/GreaterEqual_2GreaterEqualrnn/while/Identityrnn/while/GreaterEqual/Enter*
T0*
_output_shapes
:

rnn/while/Select_2Selectrnn/while/GreaterEqual_2rnn/while/Identity_3Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_2*
T0*
_output_shapes

:d*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_2

rnn/while/GreaterEqual_3GreaterEqualrnn/while/Identityrnn/while/GreaterEqual/Enter*
T0*
_output_shapes
:

rnn/while/Select_3Selectrnn/while/GreaterEqual_3rnn/while/Identity_4Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add_1*
T0*
_output_shapes

:d*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add_1

rnn/while/GreaterEqual_4GreaterEqualrnn/while/Identityrnn/while/GreaterEqual/Enter*
T0*
_output_shapes
:

rnn/while/Select_4Selectrnn/while/GreaterEqual_4rnn/while/Identity_5Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2*
T0*
_output_shapes

:d*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2

3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
_output_shapes
:*$

frame_namernn/while/rnn/while/*
parallel_iterations *
T0*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2*
is_constant(
Ž
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identityrnn/while/Selectrnn/while/Identity_1*
T0*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2
f
rnn/while/add/yConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Z
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
T0*
_output_shapes
: 
X
rnn/while/NextIterationNextIterationrnn/while/add*
T0*
_output_shapes
: 
z
rnn/while/NextIteration_1NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
g
rnn/while/NextIteration_2NextIterationrnn/while/Select_1*
T0*
_output_shapes

:d
g
rnn/while/NextIteration_3NextIterationrnn/while/Select_2*
T0*
_output_shapes

:d
g
rnn/while/NextIteration_4NextIterationrnn/while/Select_3*
T0*
_output_shapes

:d
g
rnn/while/NextIteration_5NextIterationrnn/while/Select_4*
T0*
_output_shapes

:d
I
rnn/while/ExitExitrnn/while/Switch*
T0*
_output_shapes
: 
M
rnn/while/Exit_1Exitrnn/while/Switch_1*
T0*
_output_shapes
: 
U
rnn/while/Exit_2Exitrnn/while/Switch_2*
T0*
_output_shapes

:d
U
rnn/while/Exit_3Exitrnn/while/Switch_3*
T0*
_output_shapes

:d
U
rnn/while/Exit_4Exitrnn/while/Switch_4*
T0*
_output_shapes

:d
U
rnn/while/Exit_5Exitrnn/while/Switch_5*
T0*
_output_shapes

:d

&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_1*
_output_shapes
: *"
_class
loc:@rnn/TensorArray

 rnn/TensorArrayStack/range/startConst*
value	B : *
_output_shapes
: *
dtype0*"
_class
loc:@rnn/TensorArray

 rnn/TensorArrayStack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0*"
_class
loc:@rnn/TensorArray
ä
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*"
_class
loc:@rnn/TensorArray
ů
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_1*
element_shape
:d*+
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
dtype0*"
_class
loc:@rnn/TensorArray
U
rnn/Const_5Const*
valueB:d*
_output_shapes
:*
dtype0
J
rnn/RankConst*
value	B :*
_output_shapes
: *
dtype0
Q
rnn/range/startConst*
value	B :*
_output_shapes
: *
dtype0
Q
rnn/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
f
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0*
_output_shapes
:
f
rnn/concat_1/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
S
rnn/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0

rnn/concat_1ConcatV2rnn/concat_1/values_0	rnn/rangernn/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N

rnn/transpose	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
Tperm0
R
ShapeShapernn/transpose*
T0*
_output_shapes
:*
out_type0
]
strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
_
strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
_
strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ů
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
_output_shapes
: *
new_axis_mask *

begin_mask *
ellipsis_mask *
T0*
shrink_axis_mask*
end_mask 
O
range_1/startConst*
value	B : *
_output_shapes
: *
dtype0
O
range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
n
range_1Rangerange_1/startstrided_slicerange_1/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
G
sub/yConst*
value	B :*
_output_shapes
: *
dtype0
A
subSub	Reshape_1sub/y*
T0*
_output_shapes
:
Y
stackPackrange_1sub*

axis*
T0*
_output_shapes

:*
N
i
GatherNdGatherNdrnn/transposestack*
Tparams0*
_output_shapes

:d*
Tindices0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"d      *
_output_shapes
:*
_class
loc:@dense/kernel*
dtype0

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *B[xž*
_output_shapes
: *
_class
loc:@dense/kernel*
dtype0

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *B[x>*
_output_shapes
: *
_class
loc:@dense/kernel*
dtype0
ć
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*

seed*
seed2ł*
dtype0*
_output_shapes

:d*
T0*
_class
loc:@dense/kernel
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@dense/kernel
ŕ
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:d*
_class
loc:@dense/kernel
Ň
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:d*
_class
loc:@dense/kernel
Ą
dense/kernel
VariableV2*
shared_name *
shape
:d*
_output_shapes

:d*
	container *
_class
loc:@dense/kernel*
dtype0
Ç
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense/kernel*
_output_shapes

:d
u
dense/kernel/readIdentitydense/kernel*
T0*
_output_shapes

:d*
_class
loc:@dense/kernel

dense/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
_class
loc:@dense/bias*
dtype0


dense/bias
VariableV2*
shared_name *
shape:*
_output_shapes
:*
	container *
_class
loc:@dense/bias*
dtype0
˛
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
T0*
_output_shapes
:*
_class
loc:@dense/bias

dense/MatMulMatMulGatherNddense/kernel/read*
transpose_a( *
T0*
_output_shapes

:*
transpose_b( 
w
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
_output_shapes

:*
data_formatNHWC
J
SoftmaxSoftmaxdense/BiasAdd*
T0*
_output_shapes

:
R
ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
o
ArgMaxArgMaxSoftmaxArgMax/dimension*
T0*

Tidx0*
_output_shapes
:*
output_type0	
Q
softmax_tensorSoftmaxdense/BiasAdd*
T0*
_output_shapes

:
P

save/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_7f94f6d13fbd449d84aca6eaf4cb7ab8/part*
_output_shapes
: *
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
Q
save/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
\
save/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Â
save/SaveV2/tensor_namesConst*ő
valueëBčB
dense/biasBdense/kernelBembedding/WBglobal_stepB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes
:*
dtype0
s
save/SaveV2/shape_and_slicesConst*#
valueBB B B B B B B B *
_output_shapes
:*
dtype0
Ű
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices
dense/biasdense/kernelembedding/Wglobal_step(rnn/multi_rnn_cell/cell_0/lstm_cell/bias*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel(rnn/multi_rnn_cell/cell_1/lstm_cell/bias*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtypes

2	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
_output_shapes
:*
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
n
save/RestoreV2/tensor_namesConst*
valueBB
dense/bias*
_output_shapes
:*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssign
dense/biassave/RestoreV2*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes
:
r
save/RestoreV2_1/tensor_namesConst*!
valueBBdense/kernel*
_output_shapes
:*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
Ş
save/Assign_1Assigndense/kernelsave/RestoreV2_1*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense/kernel*
_output_shapes

:d
q
save/RestoreV2_2/tensor_namesConst* 
valueBBembedding/W*
_output_shapes
:*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
Š
save/Assign_2Assignembedding/Wsave/RestoreV2_2*
use_locking(*
T0*
validate_shape(*
_class
loc:@embedding/W*
_output_shapes
:	Ŕ
q
save/RestoreV2_3/tensor_namesConst* 
valueBBglobal_step*
_output_shapes
:*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2	
 
save/Assign_3Assignglobal_stepsave/RestoreV2_3*
use_locking(*
T0	*
validate_shape(*
_class
loc:@global_step*
_output_shapes
: 

save/RestoreV2_4/tensor_namesConst*=
value4B2B(rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
:*
dtype0
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
ß
save/Assign_4Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave/RestoreV2_4*
use_locking(*
T0*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:

save/RestoreV2_5/tensor_namesConst*?
value6B4B*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes
:*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
č
save/Assign_5Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave/RestoreV2_5*
use_locking(*
T0*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:


save/RestoreV2_6/tensor_namesConst*=
value4B2B(rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes
:*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
ß
save/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave/RestoreV2_6*
use_locking(*
T0*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes	
:

save/RestoreV2_7/tensor_namesConst*?
value6B4B*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes
:*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
č
save/Assign_7Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave/RestoreV2_7*
use_locking(*
T0*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel* 
_output_shapes
:
Č

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp
+

group_depsNoOp^init^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_e8f643a98f1b49c1b7bebe52ffb9b636/part*
_output_shapes
: *
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_1/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_1/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Ä
save_1/SaveV2/tensor_namesConst*ő
valueëBčB
dense/biasBdense/kernelBembedding/WBglobal_stepB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes
:*
dtype0
u
save_1/SaveV2/shape_and_slicesConst*#
valueBB B B B B B B B *
_output_shapes
:*
dtype0
ă
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices
dense/biasdense/kernelembedding/Wglobal_step(rnn/multi_rnn_cell/cell_0/lstm_cell/bias*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel(rnn/multi_rnn_cell/cell_1/lstm_cell/bias*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtypes

2	

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
Ł
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
T0*
_output_shapes
:*
N

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
T0*
_output_shapes
: 
p
save_1/RestoreV2/tensor_namesConst*
valueBB
dense/bias*
_output_shapes
:*
dtype0
j
!save_1/RestoreV2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
˘
save_1/AssignAssign
dense/biassave_1/RestoreV2*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes
:
t
save_1/RestoreV2_1/tensor_namesConst*!
valueBBdense/kernel*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_1/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
Ž
save_1/Assign_1Assigndense/kernelsave_1/RestoreV2_1*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense/kernel*
_output_shapes

:d
s
save_1/RestoreV2_2/tensor_namesConst* 
valueBBembedding/W*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
­
save_1/Assign_2Assignembedding/Wsave_1/RestoreV2_2*
use_locking(*
T0*
validate_shape(*
_class
loc:@embedding/W*
_output_shapes
:	Ŕ
s
save_1/RestoreV2_3/tensor_namesConst* 
valueBBglobal_step*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_3/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2	
¤
save_1/Assign_3Assignglobal_stepsave_1/RestoreV2_3*
use_locking(*
T0	*
validate_shape(*
_class
loc:@global_step*
_output_shapes
: 

save_1/RestoreV2_4/tensor_namesConst*=
value4B2B(rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
ă
save_1/Assign_4Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_1/RestoreV2_4*
use_locking(*
T0*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:

save_1/RestoreV2_5/tensor_namesConst*?
value6B4B*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_5/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
ě
save_1/Assign_5Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_1/RestoreV2_5*
use_locking(*
T0*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:


save_1/RestoreV2_6/tensor_namesConst*=
value4B2B(rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_6/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
ă
save_1/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_1/RestoreV2_6*
use_locking(*
T0*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes	
:

save_1/RestoreV2_7/tensor_namesConst*?
value6B4B*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_7/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
ě
save_1/Assign_7Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_1/RestoreV2_7*
use_locking(*
T0*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel* 
_output_shapes
:
Č
Ş
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8" 
legacy_init_op


group_deps"Ĺ
	variablesˇ´
7
global_step:0global_step/Assignglobal_step/read:0
7
embedding/W:0embedding/W/Assignembedding/W/read:0

,rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0

*rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0

,rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0

*rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0"
trainable_variablesţű
7
embedding/W:0embedding/W/Assignembedding/W/read:0

,rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0

*rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0

,rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0

*rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0" 
global_step

global_step:0"Á?
while_contextŻ?Ź?
î	
map/while/map/while/
*map/while/LoopCond:02map/while/Merge:0:map/while/Identity:0Bmap/while/Exit:0Bmap/while/Exit_1:0JÄ
map/TensorArray:0
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map/TensorArray_1:0
map/strided_slice:0
map/while/Enter:0
map/while/Enter_1:0
map/while/Exit:0
map/while/Exit_1:0
map/while/Identity:0
map/while/Identity_1:0
map/while/Less/Enter:0
map/while/Less:0
map/while/LoopCond:0
map/while/Merge:0
map/while/Merge:1
map/while/Merge_1:0
map/while/Merge_1:1
map/while/NextIteration:0
map/while/NextIteration_1:0
map/while/PyFunc:0
map/while/Switch:0
map/while/Switch:1
map/while/Switch_1:0
map/while/Switch_1:1
#map/while/TensorArrayReadV3/Enter:0
%map/while/TensorArrayReadV3/Enter_1:0
map/while/TensorArrayReadV3:0
5map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/map/while/TensorArrayWrite/TensorArrayWriteV3:0
map/while/add/y:0
map/while/add:08
map/TensorArray:0#map/while/TensorArrayReadV3/Enter:0L
map/TensorArray_1:05map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0i
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%map/while/TensorArrayReadV3/Enter_1:0-
map/strided_slice:0map/while/Less/Enter:0Rmap/while/Enter:0Rmap/while/Enter_1:0
¸5
rnn/while/rnn/while/ *rnn/while/LoopCond:02rnn/while/Merge:0:rnn/while/Identity:0Brnn/while/Exit:0Brnn/while/Exit_1:0Brnn/while/Exit_2:0Brnn/while/Exit_3:0Brnn/while/Exit_4:0Brnn/while/Exit_5:0Ję2
rnn/CheckSeqLen:0
rnn/TensorArray:0
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn/TensorArray_1:0
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0
rnn/strided_slice:0
rnn/while/Enter:0
rnn/while/Enter_1:0
rnn/while/Enter_2:0
rnn/while/Enter_3:0
rnn/while/Enter_4:0
rnn/while/Enter_5:0
rnn/while/Exit:0
rnn/while/Exit_1:0
rnn/while/Exit_2:0
rnn/while/Exit_3:0
rnn/while/Exit_4:0
rnn/while/Exit_5:0
rnn/while/GreaterEqual/Enter:0
rnn/while/GreaterEqual:0
rnn/while/GreaterEqual_1:0
rnn/while/GreaterEqual_2:0
rnn/while/GreaterEqual_3:0
rnn/while/GreaterEqual_4:0
rnn/while/Identity:0
rnn/while/Identity_1:0
rnn/while/Identity_2:0
rnn/while/Identity_3:0
rnn/while/Identity_4:0
rnn/while/Identity_5:0
rnn/while/Less/Enter:0
rnn/while/Less:0
rnn/while/LoopCond:0
rnn/while/Merge:0
rnn/while/Merge:1
rnn/while/Merge_1:0
rnn/while/Merge_1:1
rnn/while/Merge_2:0
rnn/while/Merge_2:1
rnn/while/Merge_3:0
rnn/while/Merge_3:1
rnn/while/Merge_4:0
rnn/while/Merge_4:1
rnn/while/Merge_5:0
rnn/while/Merge_5:1
rnn/while/NextIteration:0
rnn/while/NextIteration_1:0
rnn/while/NextIteration_2:0
rnn/while/NextIteration_3:0
rnn/while/NextIteration_4:0
rnn/while/NextIteration_5:0
rnn/while/Select/Enter:0
rnn/while/Select:0
rnn/while/Select_1:0
rnn/while/Select_2:0
rnn/while/Select_3:0
rnn/while/Select_4:0
rnn/while/Switch:0
rnn/while/Switch:1
rnn/while/Switch_1:0
rnn/while/Switch_1:1
rnn/while/Switch_2:0
rnn/while/Switch_2:1
rnn/while/Switch_3:0
rnn/while/Switch_3:1
rnn/while/Switch_4:0
rnn/while/Switch_4:1
rnn/while/Switch_5:0
rnn/while/Switch_5:1
#rnn/while/TensorArrayReadV3/Enter:0
%rnn/while/TensorArrayReadV3/Enter_1:0
rnn/while/TensorArrayReadV3:0
5rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn/while/add/y:0
rnn/while/add:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Const:0
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoid:0
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoid_1:0
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Sigmoid_2:0
Ernn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Tanh:0
Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/Tanh_1:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add/y:0
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/add_1:0
Xrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/BiasAdd/Enter:0
Rrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/BiasAdd:0
Wrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/MatMul/Enter:0
Qrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/MatMul:0
Vrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/concat/axis:0
Qrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/concat:0
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_1:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/mul_2:0
Prnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split/split_dim:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split:1
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split:2
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/split:3
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Const:0
Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoid:0
Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoid_1:0
Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Sigmoid_2:0
Ernn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Tanh:0
Grnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/Tanh_1:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add/y:0
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/add_1:0
Xrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/BiasAdd/Enter:0
Rrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/BiasAdd:0
Wrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/MatMul/Enter:0
Qrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/MatMul:0
Vrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/concat/axis:0
Qrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/concat:0
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_1:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/mul_2:0
Prnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split/split_dim:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split:1
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split:2
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/split:3
rnn/zeros:0
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0Wrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/MatMul/Enter:0
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0Xrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/lstm_cell/lstm_cell/BiasAdd/Enter:0'
rnn/zeros:0rnn/while/Select/Enter:0:
rnn/TensorArray_1:0#rnn/while/TensorArrayReadV3/Enter:0
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0Xrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/BiasAdd/Enter:03
rnn/CheckSeqLen:0rnn/while/GreaterEqual/Enter:0J
rnn/TensorArray:05rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0i
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%rnn/while/TensorArrayReadV3/Enter_1:0-
rnn/strided_slice:0rnn/while/Less/Enter:0
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0Wrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/lstm_cell/lstm_cell/lstm_cell/MatMul/Enter:0Rrnn/while/Enter:0Rrnn/while/Enter_1:0Rrnn/while/Enter_2:0Rrnn/while/Enter_3:0Rrnn/while/Enter_4:0Rrnn/while/Enter_5:0*Š

prediction
,
sequence 
Placeholder:0˙˙˙˙˙˙˙˙˙
classes
ArgMax:0	/
probabilities
softmax_tensor:0tensorflow/serving/predict*Ž
serving_default
,
sequence 
Placeholder:0˙˙˙˙˙˙˙˙˙
classes
ArgMax:0	/
probabilities
softmax_tensor:0tensorflow/serving/predict