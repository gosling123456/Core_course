## 1.在 StudentInfo 数据库中创建 student 表和 grade 表。

Student 表的内容

| 字段名   | 字段描述 | 数据类型    | 主键 | 非空 | 唯一 | 自增 |
| -------- | -------- | ----------- | ---- | ---- | ---- | ---- |
| num      | 学号     | INT(10)     | 是   | 是   | 是   | 否   |
| name     | 姓名     | VARCHAR(20) | 否   | 是   | 否   | 否   |
| sex      | 性别     | VARCHAR(4)  | 否   | 是   | 否   | 否   |
| birthday | 出生日期 | DATETIME    | 否   | 否   | 否   | 否   |
| address  | 家庭住址 | VARCHAR(50) | 否   | 否   | 否   | 否   |

Grade 表的内容

| 字段名 | 字段描述 | 数据类型    | 主键 | 非空 | 唯一 | 自增 |
| ------ | -------- | ----------- | ---- | ---- | ---- | ---- |
| id     | 编号     | INT(10)     | 是   | 是   | 是   | 否   |
| Course | 课程名   | VARCHAR(10) | 否   | 是   | 否   | 否   |
| S_num  | 学号     | INT(10)     | 否   | 是   | 否   | 否   |
| grade  | 成绩     | VARCHAR(4)  | 否   | 否   | 否   | 否   |

创建学生表

```sql
use studentInfo
CREATE TABLE student(
num INt(10) NOT NULL UNIQUE PRIMARY KEY COMMENT '学号',
name VARCHAR(20) NOT NULL COMMENT '姓名',
sex VARCHAR(4) NOT NULL COMMENT '性别',
birthday DATETIME COMMENT '出生日期',
address VARCHAR(50) COMMENT '家庭住址'
);
```

创建成绩表

```sql

CREATE TABLE grade(
id INt(10) NOT NULL UNIQUE PRIMARY KEY COMMENT '编号',
Course VARCHAR(20) NOT NULL COMMENT '课程名',
S_num INT(10) NOT NULL COMMENT '学号',
grade VARCHAR(4) COMMENT '成绩'
);
```

增加字段外键删除外键

```sql
#声明外键
constraint 外键名 foreign key(要作为外键字段名) references 主表名(主表中关联的字段)
#/*给存在的表加外键删除外键*/
#alter table 表名 add constraint FK_ID foreign key(你的外键字段名) REFERENCES 外表表名(对应的表的主键字段名);
alter table grade add foreign key S_num references student;
alter table grade add constraint FK_ID foreign key S_num REFERENCES student;
1.主键约束
添加:alter table  table_name add primary key (字段)
删除:alter table table_name drop primary key
2.非空约束
添加:alter  table table_name modify 列名 数据类型  not null 
删除:alter table table_name modify 列名 数据类型 null
3.唯一约束
添加:alter table table_name add unique 约束名（字段）
删除:alter table table_name drop key 约束名
4.自动增长
添加:alter table table_name  modify 列名 int  auto_increment
删除:alter table table_name modify 列名 int  
5.外键约束
添加:alter table table_name add constraint 约束名 foreign key(外键列) 
references 主键表（主键列）
alter table grade add constraint fk foreign key tt  references student;
删除:
第一步:删除外键
alter table table_name drop foreign key 约束名
第二步:删除索引
alter  table table_name drop  index 索引名
[^1]: 
约束名和索引名一样
6.默认值
添加:alter table table_name alter 列名  set default '值'
删除:alter table table_name alter 列名  drop default
```

查看表

```sql
SHOW TABLES;
```

相关操作

表创建成功后，查看两个表的结构。然后按下列要求进行表操作，写出相关的命令行：
 1．将 grade 表的 course 字段的数据类型改为 VARCHAR(20)
 2．将 s_num 字段的位置改到 course 字段的前面。
 3．将 grade 字段改名为 score。
 4．删除 grade 表的外键约束。
 5．将 grade 表的存储引擎更改为 MyISAM 类型。
 6．将 student 表的 address 字段删除。
 7．在 student 表中增加名为 phone 的字段。
 8．将 grade 表改名为 gradeInfo。
 9．删除 student 表。

```SQl
#ALTER TABLE <表名> MODIFY <字段名> <数据类型>;
ALTER TABLE grade MODIFY Course VARCHAR(20);
DESC Course;
#ALTER TABLE <表名> MODIFY <字段名> <数据类型> [FIRST|AFTER 字段名2];
ALTER TABLE grade MODIFY S_num INT(10) FIRST;#添加到第一个字段
ALTER TABLE grade MODIFY Course varchar(20) NOT NULL AFTER S_num;
#ALTER TABLE <表名> change <原字段名> <新字段名> <列类型> <列属性>;
ALTER TABLE grade CHANGE grade score varchar(4);
#ALTER TABLE <表名> ENGINE=<引擎>;
alter table grade engine=MyISAM;
#ALTER TABLE <表名> DROP <字段名>
ALTER TABLE student DROP address;
#alter table <表名> add <字段名> <数据类型> [FIRST|AFTER 字段名2]; 可指定位置
alter table student add phone INT(10);
#alter table <表名> rename [to|as] <新表名>;
alter table grade rename to gradeInfo;
#drop table <表名>
drop table student;
```



## 2.教材P83设计用户表

创建用户表

```sql
create table user(
id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '用户id',
username VARCHAR(20) UNIQUE NOT NULL COMMENT '用户名',
mobile CHAR(11) NOT NULL COMMENT '手机号码',
gender ENUM('男','女','保密') NOT NULL COMMENT '性别',
reg_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
level TINYINT UNSIGNED NOT NULL COMMENT '会员等级'
) DEFAULT CHARSET=utf8;

```

添加测试记录

```SQL
INSERT INTO user VALUES(
NULL, '小明',
'12313456284', '男',
'2022-11-03 15:24:16', 1
);
```

查询用户表中记录

```sql
SELECT * FROM user;
```

